from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import tls_client

class Crash:
    def get_games(self):    
        t = tls_client.Session(client_identifier="Chrome_104")   
        headers = {
            'authority': 'api.bloxflip.com',
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'en-US,en;q=0.9,ru;q=0.8',
            'if-modified-since': 'Sun, 27 Aug 2023 03:29:12 GMT',
            'origin': 'https://bloxflip.com',
            'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
        }

        response = t.get('https://api.bloxflip.com/games/crash', headers=headers)
        return [response.json()['history'][x]['crashPoint'] for x in range(self.amt_of_games_to_fetch + 1)]
    def __init__(self) -> None:
        # amount of games to fetch from 
        self.amt_of_games_to_fetch = 17
        # makes it print the cat boost learning turn to false if you dont want this
        self.output_from_cat_boost = True
        self.seq = self.get_games()
        self.new = self.seq[0]
        self.nn = CatBoostRegressor(
            verbose=self.output_from_cat_boost
        )
    def predict(self):  
        # x matrix | input vals 
        x = np.array(self.seq).reshape(-1,1)
        y = np.array(self.seq)
        # fits the x,y on the catboostregressor
        self.nn.fit(x,y)
        gay = self.nn.predict([[self.new]])
        # rounds it by 2 n-digits
        print("Prediction: " + str(round(gay[0],2)))

c = Crash()
c.predict()
