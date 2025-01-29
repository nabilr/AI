**Python model for algorithmic trading** using a **rational agent approach**. This model follows the principles of a **rational AI agent** by:

- **Perceiving** stock market data  
- **Understanding** trends using technical indicators  
- **Evaluating** potential trades  
- **Executing** optimal buy/sell decisions  
- **Learning** from past trades  

The model uses **historical stock price data** and applies a **simple moving average (SMA) crossover strategy** to determine buy/sell signals.

---

### **🔹 Steps in the Code**
1. **Fetch stock data** using `yfinance`
2. **Compute technical indicators** (Short & Long Moving Averages)
3. **Decide on trades** based on crossover strategy
4. **Simulate trading** and track performance
5. **Evaluate the model's success**

---

### **🔹 Install Required Libraries**
```bash
pip install yfinance pandas numpy matplotlib
```

---

### **🔹 Python Code for Algorithmic Trading Rational Agent**
```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define a rational trading agent class
class TradingAgent:
    def __init__(self, stock_symbol, short_window=50, long_window=200, initial_balance=10000):
        self.stock_symbol = stock_symbol
        self.short_window = short_window
        self.long_window = long_window
        self.balance = initial_balance  # Starting balance in $
        self.shares = 0  # No shares owned initially
        self.transaction_history = []  # Stores buy/sell transactions

    def get_stock_data(self):
        """Fetches historical stock price data."""
        stock = yf.Ticker(self.stock_symbol)
        df = stock.history(period="2y")  # Get 2 years of data
        df['Short_MA'] = df['Close'].rolling(window=self.short_window).mean()
        df['Long_MA'] = df['Close'].rolling(window=self.long_window).mean()
        df.dropna(inplace=True)  # Remove NaN values
        return df

    def make_decision(self, df):
        """Evaluates trading decisions based on moving average crossover strategy."""
        df['Signal'] = 0  # Default no action
        df.loc[df['Short_MA'] > df['Long_MA'], 'Signal'] = 1  # Buy Signal
        df.loc[df['Short_MA'] < df['Long_MA'], 'Signal'] = -1  # Sell Signal
        return df

    def execute_trades(self, df):
        """Executes trades based on the decision strategy."""
        for date, row in df.iterrows():
            price = row['Close']
            signal = row['Signal']

            if signal == 1 and self.balance >= price:  # Buy Signal
                shares_to_buy = self.balance // price
                self.shares += shares_to_buy
                self.balance -= shares_to_buy * price
                self.transaction_history.append((date, "BUY", price, self.shares, self.balance))

            elif signal == -1 and self.shares > 0:  # Sell Signal
                self.balance += self.shares * price
                self.transaction_history.append((date, "SELL", price, self.shares, self.balance))
                self.shares = 0

    def backtest(self):
        """Backtests the strategy over historical data and plots performance."""
        df = self.get_stock_data()
        df = self.make_decision(df)
        self.execute_trades(df)

        # Plot stock price and buy/sell points
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label="Stock Price", color="blue", alpha=0.7)
        plt.plot(df.index, df['Short_MA'], label="50-Day MA", color="green", linestyle="--")
        plt.plot(df.index, df['Long_MA'], label="200-Day MA", color="red", linestyle="--")

        # Mark buy and sell signals
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]
        plt.scatter(buy_signals.index, buy_signals['Close'], label="Buy Signal", marker="^", color="green", s=100)
        plt.scatter(sell_signals.index, sell_signals['Close'], label="Sell Signal", marker="v", color="red", s=100)

        plt.title(f"Trading Strategy for {self.stock_symbol}")
        plt.legend()
        plt.show()

        # Final report
        print(f"Final Balance: ${self.balance:.2f}")
        print(f"Total Transactions: {len(self.transaction_history)}")
        return self.transaction_history

# Run the rational trading agent
trading_agent = TradingAgent(stock_symbol="AAPL")  # You can replace with any stock symbol
trade_history = trading_agent.backtest()
```

---

### **🔹 How This Code Implements Rational Agent Principles**
| **Rational Agent Step**  | **Implementation in Code** |
|------------------------|--------------------------------|
| **Perceiving the Environment** | Fetches stock data using `yfinance`. |
| **Understanding Market State** | Computes moving averages to identify trends. |
| **Evaluating Possible Trades** | Compares short & long MA crossovers. |
| **Selecting Best Action** | Buys when short MA > long MA, sells when opposite. |
| **Executing and Learning** | Trades based on signals and backtests performance. |

---

### **🔹 Expected Output**
- **A chart showing stock price and trade signals (buy/sell points)**  
- **Final trading balance and total transactions executed**

---

### **🔹 Key Takeaways**
✔ Uses **real stock data** to make decisions.  
✔ Implements **a rational decision-making strategy** (moving average crossover).  
✔ **Backtests** performance over historical data.  
✔ Can be **expanded** to include AI-based predictions (e.g., deep learning models).  




### **Extending the Trading Agent with AI-Based Predictions (LSTM Model) 🤖📈**
To enhance the rational trading agent, we will **replace the Moving Average strategy** with a **Deep Learning model (LSTM - Long Short-Term Memory Neural Network)**. The LSTM model will **predict future stock prices** based on historical data, allowing the agent to make **smarter trading decisions**.

---

### **🔹 Steps in the AI-Powered Trading Model**
1. **Collect stock data** using `yfinance`
2. **Preprocess the data** (scaling, feature engineering)
3. **Train an LSTM model** to predict future prices
4. **Make trading decisions** based on predicted price trends
5. **Backtest the strategy** to evaluate performance

---

### **🔹 Install Required Libraries**
```bash
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow
```

---

### **🔹 AI-Based Trading Agent Code (LSTM Model)**
```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define AI-Powered Trading Agent
class AITradingAgent:
    def __init__(self, stock_symbol, lookback=50, initial_balance=10000):
        self.stock_symbol = stock_symbol
        self.lookback = lookback  # Number of past days to use for prediction
        self.balance = initial_balance
        self.shares = 0
        self.transaction_history = []
    
    def get_stock_data(self):
        """Fetch historical stock data."""
        stock = yf.Ticker(self.stock_symbol)
        df = stock.history(period="5y")  # Get 5 years of data
        df = df[['Close']]  # Use only closing prices
        return df
    
    def preprocess_data(self, df):
        """Normalize and create input sequences for LSTM."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(df)

        X, y = [], []
        for i in range(self.lookback, len(df_scaled) - 1):
            X.append(df_scaled[i - self.lookback:i, 0])
            y.append(df_scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        return X.reshape((X.shape[0], X.shape[1], 1)), y, scaler
    
    def build_lstm_model(self):
        """Define and compile the LSTM model."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_model(self, X_train, y_train):
        """Train the LSTM model."""
        model = self.build_lstm_model()
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)
        return model
    
    def make_predictions(self, model, X_test, scaler):
        """Use the trained model to predict future stock prices."""
        predictions = model.predict(X_test)
        return scaler.inverse_transform(predictions.reshape(-1, 1))
    
    def execute_trades(self, df, predictions):
        """Trade based on AI predictions."""
        for i in range(len(predictions) - 1):
            predicted_price = predictions[i]
            current_price = df.iloc[i + self.lookback]['Close']

            if predicted_price > current_price and self.balance >= current_price:  # Buy signal
                shares_to_buy = self.balance // current_price
                self.shares += shares_to_buy
                self.balance -= shares_to_buy * current_price
                self.transaction_history.append((df.index[i + self.lookback], "BUY", current_price, self.shares, self.balance))

            elif predicted_price < current_price and self.shares > 0:  # Sell signal
                self.balance += self.shares * current_price
                self.transaction_history.append((df.index[i + self.lookback], "SELL", current_price, self.shares, self.balance))
                self.shares = 0

    def backtest(self):
        """Runs the AI trading strategy and evaluates its performance."""
        df = self.get_stock_data()
        X, y, scaler = self.preprocess_data(df)
        
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test = X[train_size:]

        # Train LSTM Model
        model = self.train_model(X_train, y_train)
        
        # Predict Future Prices
        predictions = self.make_predictions(model, X_test, scaler)

        # Execute Trades
        self.execute_trades(df, predictions)

        # Plot Results
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[-len(predictions):], df['Close'][-len(predictions):], label="Actual Prices", color="blue")
        plt.plot(df.index[-len(predictions):], predictions, label="Predicted Prices", color="orange")
        plt.title(f"AI Trading Strategy for {self.stock_symbol}")
        plt.legend()
        plt.show()

        # Final Report
        print(f"Final Balance: ${self.balance:.2f}")
        print(f"Total Transactions: {len(self.transaction_history)}")
        return self.transaction_history

# Run AI Trading Agent
ai_trading_agent = AITradingAgent(stock_symbol="AAPL")  # Replace with any stock symbol
trade_history = ai_trading_agent.backtest()
```

---

### **🔹 How This Model Works as a Rational Agent**
| **Rational Agent Step**  | **Implementation in Code** |
|------------------------|--------------------------------|
| **Perceiving the Environment** | Fetches real stock data using `yfinance`. |
| **Understanding Market State** | Uses LSTM deep learning to predict price trends. |
| **Evaluating Possible Trades** | Compares predicted vs. current stock price. |
| **Selecting Best Action** | Buys if price is expected to rise, sells if it will drop. |
| **Executing and Learning** | Simulates trading and adjusts based on predictions. |

---

### **🔹 Expected Output**
- **Graph comparing actual vs. predicted stock prices** 📊  
- **Trading signals based on AI predictions** ✅  
- **Final balance after executing trades** 💰  

---

### **🔹 Key Features & Improvements**
✔ **Uses Deep Learning (LSTM)** instead of basic moving averages.  
✔ **Predicts future prices** instead of relying on fixed rules.  
✔ **Trades automatically based on AI insights**.  
✔ **Self-improving** – LSTM learns from past market data.  

---

### **🔹 Future Enhancements**
🔹 Add **Reinforcement Learning (RL)** to optimize trading strategies further.  
🔹 Integrate **real-time stock market data** for live trading.  
🔹 Improve model by using **Sentiment Analysis on financial news**.  

---

### **Conclusion**
This **AI-powered rational agent** enhances algorithmic trading by using **deep learning (LSTMs) to predict stock trends** and **execute intelligent trades**. Unlike traditional rule-based models, this **learns from historical data and adapts to changing market conditions**, making it a **true rational agent**.  

### **AI-Powered Trading Agent with Reinforcement Learning (Deep Q-Learning) 🤖📈**
To further enhance our **rational trading agent**, we will now integrate **Reinforcement Learning (RL)** using **Deep Q-Learning (DQN)**.

---

### **🔹 Why Reinforcement Learning for Trading?**
✅ **Self-Improving** – Learns from past trading experiences.  
✅ **Adaptive Decision-Making** – Adjusts strategies based on changing market conditions.  
✅ **Maximizes Long-Term Profits** – Balances risk and reward intelligently.  
✅ **Works in Uncertain Markets** – Learns optimal actions even with market fluctuations.  

---

### **🔹 Steps in the RL-Powered Trading Model**
1. **Collect stock market data** using `yfinance`.  
2. **Define the RL environment** (states, actions, rewards).  
3. **Train the agent** using **Deep Q-Learning (DQN)**.  
4. **Test the trained agent** in a backtest simulation.  
5. **Optimize and analyze performance**.

---

### **🔹 Install Required Libraries**
```bash
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow gym
```

---

### **🔹 RL-Powered Trading Agent Code (Deep Q-Learning)**
```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import gym
from gym import spaces
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the Reinforcement Learning Environment
class StockTradingEnv(gym.Env):
    def __init__(self, stock_symbol, lookback=50, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.stock_symbol = stock_symbol
        self.lookback = lookback
        self.balance = initial_balance
        self.shares = 0
        self.transaction_history = []
        
        # Load stock data
        self.df = self.get_stock_data()
        self.current_step = self.lookback

        # Action Space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation Space: stock price, balance, shares
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.lookback + 2,), dtype=np.float32)

    def get_stock_data(self):
        """Fetches historical stock price data."""
        stock = yf.Ticker(self.stock_symbol)
        df = stock.history(period="5y")  # Get 5 years of data
        df = df[['Close']]  # Use only closing prices
        scaler = MinMaxScaler()
        df['Scaled_Close'] = scaler.fit_transform(df[['Close']])
        return df

    def reset(self):
        """Resets the environment to the initial state."""
        self.balance = 10000
        self.shares = 0
        self.current_step = self.lookback
        return self._next_observation()

    def _next_observation(self):
        """Returns the current observation (state)."""
        obs = np.array(self.df['Scaled_Close'][self.current_step - self.lookback: self.current_step])
        obs = np.append(obs, [self.balance / 10000, self.shares / 100])
        return obs.astype(np.float32)

    def step(self, action):
        """Executes a step in the environment based on the action."""
        current_price = self.df.iloc[self.current_step]['Close']
        
        if action == 1 and self.balance >= current_price:  # Buy
            shares_to_buy = self.balance // current_price
            self.shares += shares_to_buy
            self.balance -= shares_to_buy * current_price
            self.transaction_history.append((self.current_step, "BUY", current_price, self.shares, self.balance))

        elif action == 2 and self.shares > 0:  # Sell
            self.balance += self.shares * current_price
            self.transaction_history.append((self.current_step, "SELL", current_price, self.shares, self.balance))
            self.shares = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        reward = self.balance + (self.shares * current_price)  # Total net worth

        return self._next_observation(), reward, done, {}

# Define the Deep Q-Network (DQN) Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """Creates the Deep Q-Network (DQN) model."""
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation="relu"),
            Dense(64, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores experiences for training."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Chooses an action based on exploration vs. exploitation."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])  # Best action

    def train(self, batch_size=32):
        """Trains the DQN model using replay memory."""
        minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Reduce exploration over time

# Train the RL Agent
env = StockTradingEnv(stock_symbol="AAPL")  # Replace with any stock
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

episodes = 100
batch_size = 32

for e in range(episodes):
    state = env.reset()
    total_reward = 0

    for time in range(len(env.df) - env.lookback):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {e+1}/{episodes} - Final Balance: ${env.balance:.2f}")
            break

    agent.train(batch_size)

# Test the trained RL agent
env.reset()
state = env._next_observation()
done = False

while not done:
    action = agent.act(state)
    state, reward, done, _ = env.step(action)

print(f"Final Balance after RL Trading: ${env.balance:.2f}")
```

---

### **🔹 How Reinforcement Learning Improves Trading**
| **Rational Agent Step**  | **Implementation in Code** |
|------------------------|--------------------------------|
| **Perceiving the Market** | Gets stock prices via `yfinance`. |
| **Understanding Market Trends** | Uses Reinforcement Learning (DQN). |
| **Evaluating Possible Actions** | Chooses `BUY`, `SELL`, or `HOLD` based on Q-values. |
| **Maximizing Profit** | Trains over multiple episodes to refine strategy. |
| **Adapting Over Time** | Learns from past trades and improves performance. |

---

### **🔹 Next Steps**
🔹 **Use real-time trading API** (e.g., Alpaca, Binance).  
🔹 **Add stop-loss and risk management strategies**.  
🔹 **Optimize hyperparameters for better performance**.  




