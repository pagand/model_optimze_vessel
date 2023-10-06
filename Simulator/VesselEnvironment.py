import os
import pickle
import random
import gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from transformers import InformerForPrediction, InformerConfig
import tkinter as tk
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GRU_update(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=4, num_layers=1, prediction_horizon=5, device="cpu"):
        super().__init__()
        self.device = device
        self.h = prediction_horizon
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.mlp = nn.Sequential( nn.ReLU(),
                                  nn.Linear(hidden_size, 2048),
                                  nn.Dropout(0.2),
                                  nn.ReLU(),
                                  nn.Linear(2048, output_size))
        self.hx_fc = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, predicted_values, past_time_features):
        xy = torch.zeros(size=(past_time_features.shape[0], 1, self.output_size)).float().to(self.device)
        hx = past_time_features.reshape(-1, 1, self.hidden_size)
        hx = hx.permute(1, 0, 2)
        out_wp = list()
        for i in range(self.h):
            ins = torch.cat([xy, predicted_values[:, i:i+1, :]], dim=1) # x
            hx, _ = self.gru(ins, hx.contiguous())
            hx = hx.reshape(-1, 2*self.hidden_size)
            hx = self.hx_fc(hx)
            d_xy = self.mlp(hx).reshape(-1, 1, self.output_size) #control v4
            hx = hx.reshape(1, -1, self.hidden_size)
            # print("dxy", d_xy)
            xy = xy + d_xy
            # print("xy plused", xy)
            out_wp.append(xy)
        pred_wp = torch.stack(out_wp, dim=1).squeeze(2)
        return pred_wp
    
class VesselEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    # model = [tf_loc, gru_loc, tf_fc, gru_fc]
    def __init__(self, rl_data, scaler, toptrips, models_file_path, reward_type = "mimic"):

        self.rl_data = rl_data
        self.manual = False
        self.run = False
        self.done =False
        self.scale_var = False
        self.max_steps = 124
        self.trip_id = 0
        self.reward_type = reward_type
        # load best 1% trips to calculate reward1
        self.hn_top = toptrips[0]
        self.nh_top = toptrips[1]
        # set scaler
        self.scaler = scaler
        # get device
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        # load forecasting models
        self._load_model(models_file_path)
        self._set_eval()

    def _load_model(self, file_path):
        # load transformer for longitude latitude prediction
        config_loc = InformerConfig.from_pretrained("huggingface/informer-tourism-monthly", 
                prediction_length=5, context_length=24, input_size=2, num_time_features=1,
                num_dynamic_real_features = 16, num_static_real_features = 4,
                lags_sequence=[1], num_static_categorical_features=0, feature_size=27)
        self.tf_loc = InformerForPrediction(config_loc).to(self.device)
        self.tf_loc.load_state_dict(torch.load(file_path[0],
                map_location=torch.device(self.device)))

        # load transformer for fc sog prediction
        config_fc = InformerConfig.from_pretrained("huggingface/informer-tourism-monthly", 
                prediction_length=5, context_length=24, input_size=2, num_time_features=1,
                num_dynamic_real_features = 11, num_static_real_features = 4,
                lags_sequence=[1], num_static_categorical_features=0, feature_size=22)
        self.tf_fc = InformerForPrediction(config_fc).to(self.device)
        self.tf_fc.load_state_dict(torch.load(file_path[1],
                map_location=torch.device(self.device)))

        # load gru models
        self.gru_loc = GRU_update(2, hidden_size=425, output_size=2, num_layers=1, prediction_horizon=5, device=self.device).to(self.device)
        self.gru_fc = GRU_update(2, hidden_size=300, output_size=2, num_layers=1, prediction_horizon=5, device=self.device).to(self.device)
        self.gru_loc.load_state_dict(torch.load(file_path[2],
                map_location=torch.device(self.device)))
        self.gru_fc.load_state_dict(torch.load(file_path[3],
                map_location=torch.device(self.device)))
        
    # set to models eval mode
    def _set_eval(self):
        self.gru_fc.eval()
        self.gru_loc.eval()
        self.tf_fc.eval()
        self.tf_loc.eval()

        # initialize values
        self.current_step = 25
        self.reward_cum = 0
        self.reward = 0
        self.obs = np.zeros([1,19], dtype=np.float64)
        self.actions = np.zeros([1,3], dtype=np.float64)

    def _get_observation(self):
        return self.obs[-25:]
    
    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if(self.trip_id < len(self.rl_data)-1):
            self.trip_id = self.trip_id + 1
        else:
            self.trip_id = 1

        self.data = self.rl_data[self.trip_id]["observations"]
        self.current_step = 25
        self.obs = self.rl_data[self.trip_id]["observations"][0:25]
        self.actions = self.rl_data[self.trip_id]["actions"][0:25]

        # get direction and other static features
        self.direction = self.rl_data[self.trip_id]["observations"][0, 13]
        self.statics = self.rl_data[self.trip_id]["observations"][0, 12:16]
        if self.direction==1:
            self.top1 = self.hn_top
            self.goal_long, self.goal_lat = np.float64(0.9965111208024382), np.float64(0.7729570345408661)
        else:
            self.top1 = self.nh_top
            self.goal_long, self.goal_lat = np.float64(0.0023259194650222526), np.float64(0)

        # calculate the cumulative reward
        self.reward_cum = 0

        return self._get_observation(), {}

    def _take_action(self, action):
        # get actions
        speed, heading, mode = action
        heading = heading
        mode = int(mode>0.5)
        if self.current_step < self.rl_data[self.trip_id]["observations"].shape[0]:
            future_obs = self.rl_data[self.trip_id]["observations"][self.current_step].copy()
        else:
            future_obs = self.rl_data[self.trip_id]["observations"][-1].copy()
        obs = self._get_observation().copy()

        actions = self.actions[-25:].copy()

        # obs_cols = [0"Time2", 1"turn", 2"acceleration",
        #    3'change_x_factor', 4'change_y_factor', 5"distance",
        #    6'current', 7'rain', 8'snowfall',9 'wind_force',10 'wind_direc', 11"resist_ratio",
        #    12"is_weekday",13 'direction', 14"season",15"hour", 
        #    16"FC", 17"SOG", 18"LATITUDE", 19'LONGITUDE',
        #    ], 
        # action_cols = ["SPEED", "HEADING", "MODE"]
        # time_feature = ["Time2", "SPEED", "HEADING", "MODE", "turn", "acceleration",
        #     "distance", 'current', 'rain', 'snowfall', 'wind_force', 'wind_direc', "resist_ratio", 
        #     "FC", "SOG"]
        # static_categorical_feature = ["is_weekday", 'direction',"season", "hour"]
        # y_cols = ["FC2", "SOG4"]
        # dynamic_real_feature = [["Time2", "SPEED", "HEADING", "MODE", "turn", "acceleration",
        #     "change_x_factor", "change_y_factor",
        #     "distance", 'current', 'rain', 'snowfall', 'wind_force', 'wind_direc', "resist_ratio"]

        # index of features only used in the fc model
        fc_feature_index = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]

        # get model inputs
        past_time_features = np.zeros([25, 17])
        past_time_features[:, 0] = obs[:, 0]
        past_time_features[:, 1:4] = actions[:] # speed, heading, mode
        past_time_features[:, 4:15] = obs[:, 1:12]
        past_time_features[:, -2:] = obs[:, 16:18]**2

        future_time_features = past_time_features[-5:].copy()
        future_time_features[:, 0] = future_time_features[:, 0] + 5/120
        future_time_features[0, [1,2,3]] = speed, heading, mode
        future_time_features[0, 4] = heading - past_time_features[-1, 2]
        change_x = np.cos((heading+90) * np.pi / 180)
        change_y = np.sin((heading-90) * np.pi / 180)
        future_time_features[0, [13,14]] = change_x, change_y

        past_values = obs[:, -2:].copy()

        # predict fc, sog, long, lat
        fc2, sog2 = self._predict(past_time_features[:, 15:17], past_time_features[:,fc_feature_index], 
                                  future_time_features[:, fc_feature_index], self.tf_fc, self.gru_fc)
        lat, long = self._predict(past_values, past_time_features, future_time_features, 
                                  self.tf_loc, self.gru_loc)
        fc, sog = min(1, max(0, fc2)**0.5), min(1, max(0, sog2)**0.5)

        # generate next observation and update obs list
        new_observe = future_obs
        new_observe[0] = future_time_features[0, 0]
        new_observe[1:3] = future_time_features[0, 4:6]
        distance = ((self.goal_long-long)**2 + (self.goal_lat-lat)**2 )**0.5
        new_observe[3] = distance
        new_observe[-4:] = fc, sog, lat, long
        # new_observe[-4:] = future_obs[16:19]

        # append new observations and actions
        self.obs = np.append(self.obs, np.expand_dims(new_observe, 0), axis=0)
        self.actions = np.append(self.actions, np.expand_dims(action, 0), axis=0)

        return fc ,sog, lat, long

    def _predict(self, past_values, past_time_features, future_time_features, tf_model, gru_model):
        future_time_features = torch.from_numpy(np.expand_dims(future_time_features, 0)).float().to(self.device)
        past_values = torch.from_numpy(np.expand_dims(past_values, 0)).float().to(self.device)
        past_time_features = torch.from_numpy(np.expand_dims(past_time_features, 0)).float().to(self.device)
        static_real_features = torch.from_numpy(np.expand_dims(self.statics, 0)).float().to(self.device)
        past_observed_mask = torch.ones(past_values.shape).to(self.device)

        # print(past_values.shape, past_time_features.shape, future_time_features.shape)
        # make prediction
        with torch.no_grad():
            outputs = tf_model.generate(past_values=past_values, past_time_features=past_time_features,
                    static_real_features=static_real_features, past_observed_mask=past_observed_mask,
                    future_time_features=future_time_features).sequences.mean(dim=1)
            outputs = gru_model(outputs, past_time_features).detach().cpu().numpy()
        return outputs[0,0,0], outputs[0,0,1]

    def _get_reward(self, long, lat, fc):
        # reward 1 distance to the top 1
        reward1 = - ((long-self.top1.loc[self.current_step, "LONGITUDE"])**2 + (lat-self.top1.loc[self.current_step, "LATITUDE"])**2 )**0.5
        if reward1 > -0.05:
            reward1 = 0
        reward1 = reward1*10
        # reward 2 fc and done reward
        reward2 = -fc
        # reward 3 mimic reward
        if self.current_step < len(self.data):
            reward3 = - ((long-self.data[self.current_step, 19])**2 + (lat-self.data[self.current_step, 18])**2 )**0.5
        else:
            reward3 = - ((long-self.data[-1, 19])**2 + (lat-self.data[-1, 18])**2 )**0.5
        # reward 4 timeout reward
        reward4 = 0
        if self.current_step >= 100:
            reward4 = -0.1*((self.current_step-90)//10)
        # return (reward1 + reward2 + reward3 + reward4) / 4
        if self.reward_type == "mimic":
            self.reward = (reward1 + reward2 + reward3 + reward4) / 4
        elif self.reward_type == "top1":
            self.reward = (reward1+reward2+reward4) / 3
        else:
            self.reward = (reward2+reward4) /2
        
        return self.reward

    def step(self, action):
        obs= self._get_observation()
        self.current_step += 1

        fc, sog, lat, long = self._take_action(action)

        # get done and termination
        done = (((long-self.goal_long)**2 + (lat-self.goal_lat)**2) < 1e-2)
        termination =  self.current_step >= self.max_steps

        reward = self._get_reward(long, lat, fc)
        self.reward_cum = self.reward_cum + reward

        if done:
            reward = reward+1/3

        self.reward = reward
        return obs, reward, done, termination, {}


    def _inv_transform_location(self, lat, long):
        array = np.zeros([1, 12],dtype=np.float64)
        array[0, [7,8]] = lat, long
        lat, long = self.scaler.inverse_transform(array)[0,[7, 8]]
        return lat, long

    def _transform_value(self, vals, indexes):
        # transform_cols = [ 'current', 'rain', 'snowfall', "pressure", 'wind_force', "resist_ratio",
#        'FC', "LATITUDE", 'LONGITUDE', 'SOG', "DEPTH", "SPEED"]
        array = np.zeros([1, 12],dtype=np.float64)
        for i in range(len(indexes)):
            array[0, indexes[i]] = vals[i]
        transformed_val = self.scaler.transform(array)[0, [indexes]]
        return transformed_val[0]
    
    def _reset(self):
        self.ax.clear()
        self.reset()
        self.status = "Reset"
        self._update_results()
    
    # create  attribute '_next_step' for button widget
    def _next_step(self, flag=False):
        if self.manual:
            self.actions = np.append(self.actions, np.expand_dims(np.array([self.speed_var.get(), self.heading_var.get(), self.mode_var.get()]), 0), axis=0)
            
        else: # from dataset
            self.actions = np.append(self.actions, np.expand_dims(np.array(self.rl_data[self.trip_id]["actions"][self.current_step]), 0), axis=0)
        action = self.actions[-1]
        obs, reward, self.done, termination, _ = self.step(action)
        print("step: ", self.current_step, "reward: ", reward, "cumulative reward: ", self.reward_cum, self.manual)
        if self.done:
            print("Done")
        if termination:
            print("Termination")
        self.root.update()
        if not flag:
            self.status = "Done"
        self._update_results()
        
        

    def _resume(self):
        if self.run:
            self.run = False
            self.status = "Stopped"
            self._update_results()
        else:
            self.run = True
            self.status = "Running ..."
        while self.run and self.current_step < self.max_steps and not self.done:
            self.status = "Running ..."
            self._next_step(flag=True)
            self.root.update()
            # make delay
            self.root.after(1000)
        
        if self.current_step > self.max_steps:
            self.status = "Reached max steps"
        if self.done:
            self.status = "Reached goal"
        self._update_results()


    def _update_results(self):
        root = self.root
        # figures
        lat, long = self.obs[:, -2].copy(), self.obs[:, -1].copy()
        for i in range(len(lat)):
            lat[i], long[i] = self._inv_transform_location(lat[i],long[i])
        
        # ax.scatter(long_lat[:,1],long_lat[:,0],c=stw, s=1)
        self.ax.scatter(long,lat, s=1)
        self.ax.set_xlim(xmin = -124.0, xmax= -123.2)
        self.ax.set_ylim(ymin=49.15, ymax=49.45)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7,rotation=90)
        # ax.axis("off")
        plt.subplots_adjust(top=0.925,     # Further fix clipping of text in the figure
                            bottom=0.16,
                            left=0.11,
                            right=0.90,
                            hspace=0.2,
                            wspace=0.2)
        
        canvas = FigureCanvasTkAgg(self.fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, sticky=W+E+N+S)
    
        # show the current step, reward and cumulative reward in the GUI in colomn 1
        step_label = Label(root, text="Step: "+str(self.current_step))
        step_label.grid(row=2, column=0, sticky=W+E+N+S)
        reward_label = Label(root, text="Current Reward: "+str(round(self.reward, 3)))
        reward_label.grid(row=2, column=2, sticky=W+E+N+S)
        reward_cum_label = Label(root, text="Cumulative Reward: "+str(round(self.reward_cum, 3)))
        reward_cum_label.grid(row=2, column=1, sticky=W+E+N+S)
        current_engine_label = Label(root, text="Type: Manual" if self.manual else "Type: Auto")
        current_engine_label.grid(row=3, column=1, sticky=W+E+N+S)

        shows = self.obs[-1].copy()
        if self.scale_var:
            # transformed values in  the original space
            array1 = np.array([shows[6],shows[7],shows[8],0, shows[9], shows[11],shows[16], shows[18], shows[19], shows[17],0,0  ])
            array1 = array1[np.newaxis, :]
            array1 = self.scaler.inverse_transform(array1)
            array1 = array1[0,:]
            shows[6:9] = array1[0:3]
            shows[9] = array1[4]
            shows[11] = array1[5]
            shows[16:20] = [array1[6],array1[9],array1[7],array1[8]]
        # show the observation in the GUI in colomn 1
        obs_label = Label(root, text="Observations: " + \
                          "\nFC: "+str(round(shows[ 16], 3)) +\
                          "\nSOG: "+str(round(shows[17], 3)) +\
                            "\nLatitude: "+str(round(shows[18], 3)) +\
                            "\nLongitude: "+str(round(shows[ 19], 3)) +\
                          "\nTime: "+str(round(shows[ 0], 3)) + \
                          "\nTurn: "+str(round(shows[ 1], 3))  + \
                            "\nAcceleration: "+str(round(shows[ 2], 3))  + \
                            "\nDistance: "+str(round(shows[5], 3))+ \
                            "\nCurrent: "+str(round(shows[6], 3))+ \
                            "\nWind Force: "+str(round(shows[9], 3))+ \
                            "\nWind Direction: "+str(round(shows[10], 3))+ \
                            "\nResist Ratio: "+str(round(shows[11], 3))+ \
                            "\nIs Weekday: "+str(round(shows[12], 3))+ \
                            "\nSeason: "+str(round(shows[14], 3))+ \
                            "\nHour: "+str(round(shows[15], 3))+ \
                            "\nDirection: "+str(round(shows[13], 3)))
        obs_label.grid(row=0, column=3, sticky=W+E+N+S)


        #   show the actions in the GUI in colomn 1
        obs_label = Label(root, text="Applied actions ")
        obs_label.grid(row=3, column=2, sticky=W+E+N+S)
        obs_label = Label(root, text=str(round(self.actions[-1, 0], 3)))
        obs_label.grid(row=4, column=2, sticky=W+E+N+S)
        obs_label = Label(root, text=str(round(self.actions[-1, 1], 3)))
        obs_label.grid(row=5, column=2, sticky=W+E+N+S)
        obs_label = Label(root, text=str(round(self.actions[-1, 2], 3)))
        obs_label.grid(row=6, column=2, sticky=W+E+N+S)

        status_label = Label(root, text="Status: \n"+self.status)
        status_label.grid(row=2, column=3, rowspan=2, sticky=W+E+N+S)


            
            


    def render(self):
        
        #define the GUI root
        self.status = "Innitialized"
        root = tk.Tk()
        self.root  = root
        # title
        root.title("West coast vessel simulator")
        # create a canvas object and display it
        root.geometry('700x480')


        # generate the figure and plot object which will be linked to the root element
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(5.5, 3.5)

        # update the results
        self._update_results()

        # create a button widget which will be linked to the button_click function
        button1 = Button(root, text="Next Step", command=self._next_step)
        button1.grid(row=1, column=0, sticky=W+E+N+S)
        # create a button widget which will be linked to the reset function
        button2 = Button(root, text="Reset", command=self._reset)
        button2.grid(row=1, column=1, sticky=W+E+N+S)
        

        def _switchButtonState():
            if (mode_entry['state'] == tk.NORMAL):
                speed_entry['state'] = tk.DISABLED
                heading_entry['state'] = tk.DISABLED
                mode_entry['state'] = tk.DISABLED
                self.manual = False
            else:
                speed_entry['state'] = tk.NORMAL
                heading_entry['state'] = tk.NORMAL
                mode_entry['state'] = tk.NORMAL
                self.manual = True
                
        def _switchButtonState1():
            if (self.scale_var == True):
                self.scale_var = False
                self._update_results()
            else:
                self.scale_var = True
                self._update_results()

        # create button widget that change label from auto to manual and viceversal as click with the states
        button3 = tk.Button(root, text="Auto/manual",command = _switchButtonState)
        button3.grid(row=3, column=0, sticky=W+E+N+S)


        button4 = tk.Button(root, text="Resume/stop",command = self._resume)
        button4.grid(row=1, column=2, sticky=W+E+N+S)

        # change the self.scale_var to true or false as the user click on the button
        button5 = tk.Button(root, text="scaled/actual", command = _switchButtonState1)
        button5.grid(row=1, column=3, sticky=W+E+N+S)



        self.speed_var = DoubleVar()
        self.heading_var = DoubleVar()
        self.mode_var = IntVar()
        self.speed_var.set(0.5)
        self.heading_var.set(0.5)
        self.mode_var.set(0)
        speed_label = Label(root, text="Speed")
        speed_label.grid(row=4, column=0, sticky=W+E+N+S)
        speed_entry = Entry(root, textvariable=self.speed_var, state=tk.DISABLED)
        speed_entry.grid(row=4, column=1, sticky=W+E+N+S)
        heading_label = Label(root, text="Heading")
        heading_label.grid(row=5, column=0, sticky=W+E+N+S)
        heading_entry = Entry(root, textvariable=self.heading_var, state=tk.DISABLED)
        heading_entry.grid(row=5, column=1, sticky=W+E+N+S)
        mode_label = Label(root, text="Mode")
        mode_label.grid(row=6, column=0, sticky=W+E+N+S)
        mode_entry = Entry(root, textvariable=self.mode_var, state=tk.DISABLED)
        mode_entry.grid(row=6, column=1, sticky=W+E+N+S)
        

        # start the GUI event loop
        root.mainloop()


        # create a canvas object and place it in the window
        # canvas = FigureCanvasTkAgg(fig,master=root_window)
        # canvas.draw()
        # canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


# main function
def main():
    # load data
    with open('data/rl_data.pkl', 'rb') as handle:
        rl_data = pickle.load(handle)
    # load scaler
    scaler=pickle.load(open('data/minmax_scaler.pkl', 'rb'))
    # load top trips
    hn_top = pd.read_csv("data/H2N_top1.csv")
    nh_top = pd.read_csv("data/N2H_top1.csv")
    toptrips = (hn_top, nh_top)

    # load models
    # [tf_loc, tf_fc, gru_loc, gru_fc]
    file_path = (
    "data/gruloc_3_checkpoint22.pt",
    "data/gru_5_checkpoint16.pt",
    "data/gruloc_3_checkpoint22_gru.pt",
    "data/gru_5_checkpoint16_gru.pt")

    # create environment
    env = VesselEnvironment(rl_data, scaler, toptrips, file_path)
    env.reset()
    env.render()


    fc_predicted = []
    sog_predicted = []
    lat_predicted = []
    long_predicted = []
    trip_ids = []
    for i in range(2):
        # reset environment
        env.reset()
        trip_ids.append(env.trip_id)


        length = rl_data[env.trip_id]["observations"].shape[0]
        fc = np.zeros((length))
        sog =  np.zeros((length))
        lat = np.zeros((length))
        long = np.zeros((length))
        for j in range(25, length):
            action = rl_data[env.trip_id]["actions"][j]
            # action[1] = action[1]
            obs = env.step(action)[0][-1, :]
            fc[j], sog[j], lat[j], long[j] = obs[-4], obs[-3], obs[-2], obs[-1]
            # if done:
            #     break
        array1 = np.zeros((length, 12))
        array1[:, 6] = fc
        array1[:, 9] = sog
        array1[:, 7] = lat
        array1[:, 8] = long
        array1 = scaler.inverse_transform(array1)
        fc_predicted.append(array1[:, 6])
        sog_predicted.append(array1[:, 9])
        lat_predicted.append(array1[:, 7])
        long_predicted.append(array1[:, 8])
    fcs = []
    sogs = []
    longs = []
    lats = []
    # TODO why i start from 0?
    for j in range(2):
        i = trip_ids[j]
        array = np.zeros((len(rl_data[i]["observations"]), 12))
        array[:, 6] = rl_data[i]["observations"][:, -4]
        array[:, 9] = rl_data[i]["observations"][:, -3]
        array[:, 7] = rl_data[i]["observations"][:, -2]
        array[:, 8] = rl_data[i]["observations"][:, -1]
        array = scaler.inverse_transform(array)
        fcs.append(array[:, 6])
        sogs.append(array[:, 9])
        lats.append(array[:, 7])
        longs.append(array[:, 8])
    # plot
    def plot(i):
        fig = plt.figure()
        grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.3)

        ax1 = plt.subplot(grid[0, 0])
        ax2 = plt.subplot(grid[0, 1:])
        ax3 = plt.subplot(grid[1, :1])
        ax4 = plt.subplot(grid[1, 1:])


        ax1.plot(range(25, len(fc_predicted[i])), fc_predicted[i][25:], label='predictions'.format(i=2))
        ax1.plot(range(25, len(fc_predicted[i])), fcs[i][25:], label='actuals'.format(i=1))
        ax1.legend(loc='best')
        ax2.plot(range(25, len(fc_predicted[i])), long_predicted[i][25:], label='predictions'.format(i=2))
        ax2.plot(range(25, len(fc_predicted[i])), longs[i][25:], label='actuals'.format(i=1))
        ax2.legend(loc='best')
        ax3.plot(range(25, len(fc_predicted[i])), lat_predicted[i][25:], label='predictions'.format(i=2))
        ax3.plot(range(25, len(fc_predicted[i])), lats[i][25:], label='actuals'.format(i=1))
        ax3.legend(loc='best')
        ax4.plot(range(25, len(sog_predicted[i])), sog_predicted[i][25:], label='predictions'.format(i=2))
        ax4.plot(range(25, len(sog_predicted[i])), sogs[i][25:], label='actuals'.format(i=1))
        ax4.legend(loc='best')
        plt.show()
    plot(0)    
    plot(1)
    print("done")

if __name__ == "__main__":
    main()