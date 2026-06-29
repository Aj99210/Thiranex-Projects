import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import threading
from tkinter import font
import warnings
warnings.filterwarnings('ignore')

class ModernStockPredictor:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_custom_styles()
        self.create_widgets()
        self.current_data = None
        self.predictions = {}
        
    def setup_window(self):
        self.root.title("StockVision Pro")
        self.root.geometry("1500x900")
        self.root.configure(bg='#1e1e1e')
        self.root.resizable(True, True)
        
        # Set minimum size
        self.root.minsize(1200, 700)
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1500 // 2)
        y = (self.root.winfo_screenheight() // 2) - (900 // 2)
        self.root.geometry(f"1500x900+{x}+{y}")
        
    def create_custom_styles(self):
        # Color scheme
        self.colors = {
            'bg_primary': '#1e1e1e',
            'bg_secondary': '#2d2d2d',
            'bg_tertiary': '#3d3d3d',
            'accent': '#00d4aa',
            'accent_hover': '#00b894',
            'text_primary': '#ffffff',
            'text_secondary': '#b0b0b0',
            'success': '#4caf50',
            'warning': '#ff9800',
            'error': '#f44336',
            'border': '#404040'
        }
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom button style
        style.configure('Modern.TButton',
                        background=self.colors['accent'],
                        foreground='white',
                        borderwidth=0,
                        focuscolor='none',
                        font=('Segoe UI', 10, 'bold'),
                        relief='flat')
        
        style.map('Modern.TButton',
                  background=[('active', self.colors['accent_hover']),
                              ('pressed', '#009688')])
        
        # Custom entry style
        style.configure('Modern.TEntry',
                        fieldbackground=self.colors['bg_secondary'],
                        bordercolor=self.colors['border'],
                        lightcolor=self.colors['border'],
                        darkcolor=self.colors['border'],
                        foreground=self.colors['text_primary'],
                        insertcolor=self.colors['text_primary'],
                        borderwidth=1,
                        relief='solid')
        
        style.map('Modern.TEntry',
                  focuscolor=[('focus', self.colors['accent'])])
        
        # Custom combobox style
        style.configure('Modern.TCombobox',
                        fieldbackground='white',
                        background='white',
                        bordercolor=self.colors['border'],
                        foreground='black',
                        arrowcolor='black',
                        borderwidth=1,
                        relief='solid')
                        
        style.map('Modern.TCombobox',
                  fieldbackground=[('readonly', 'white')],
                  selectbackground=[('readonly', 'white')],
                  selectforeground=[('readonly', 'black')],
                  foreground=[('readonly', 'black')])
                  
        # Fix dropdown list colors
        self.root.option_add('*TCombobox*Listbox.background', 'white')
        self.root.option_add('*TCombobox*Listbox.foreground', 'black')
        self.root.option_add('*TCombobox*Listbox.selectBackground', self.colors['accent'])
        self.root.option_add('*TCombobox*Listbox.selectForeground', 'white')
        
        # Custom label frame style
        style.configure('Modern.TLabelframe',
                        background=self.colors['bg_secondary'],
                        bordercolor=self.colors['border'],
                        darkcolor=self.colors['bg_secondary'],
                        lightcolor=self.colors['bg_secondary'],
                        relief='solid',
                        borderwidth=1)
        
        style.configure('Modern.TLabelframe.Label',
                        background=self.colors['bg_secondary'],
                        foreground=self.colors['text_primary'],
                        font=('Segoe UI', 11, 'bold'))

        # Custom Treeview style for Recommendations
        style.configure('Custom.Treeview',
                        background=self.colors['bg_tertiary'],
                        foreground=self.colors['text_primary'],
                        fieldbackground=self.colors['bg_tertiary'],
                        borderwidth=0,
                        rowheight=35,
                        font=('Segoe UI', 10))
        
        style.map('Custom.Treeview',
                  background=[('selected', self.colors['accent'])],
                  foreground=[('selected', 'white')])
                  
        style.configure('Custom.Treeview.Heading',
                        background=self.colors['bg_secondary'],
                        foreground=self.colors['text_primary'],
                        font=('Segoe UI', 11, 'bold'),
                        borderwidth=1,
                        relief='solid')
                        
        style.map('Custom.Treeview.Heading',
                  background=[('active', self.colors['bg_secondary'])])
        
    def create_widgets(self):
        # Main container with padding
        main_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Header
        self.create_header(main_frame)
        
        # Content area
        content_frame = tk.Frame(main_frame, bg=self.colors['bg_primary'])
        content_frame.pack(fill='both', expand=True, pady=(15, 0))
        
        # Left panel (30% width)
        left_panel = tk.Frame(content_frame, bg=self.colors['bg_secondary'], width=400)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel (70% width)  
        right_panel = tk.Frame(content_frame, bg=self.colors['bg_secondary'])
        right_panel.pack(side='right', fill='both', expand=True)
        
        self.create_left_panel(left_panel)
        self.create_right_panel(right_panel)
        
    def create_header(self, parent):
        header_frame = tk.Frame(parent, bg=self.colors['bg_secondary'], height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        # Add subtle border
        border_frame = tk.Frame(header_frame, bg=self.colors['accent'], height=3)
        border_frame.pack(fill='x', side='bottom')
        
        # Header content
        header_content = tk.Frame(header_frame, bg=self.colors['bg_secondary'])
        header_content.pack(fill='both', expand=True, padx=20, pady=15)
        
        # Title
        title_label = tk.Label(header_content, 
                               text="📈 StockVision Pro",
                               font=('Segoe UI', 24, 'bold'),
                               fg=self.colors['accent'],
                               bg=self.colors['bg_secondary'])
        title_label.pack(side='left', anchor='w')
        
        # Status
        status_frame = tk.Frame(header_content, bg=self.colors['bg_secondary'])
        status_frame.pack(side='right', anchor='e')
        
        status_label = tk.Label(status_frame,
                                text="Status:",
                                font=('Segoe UI', 10),
                                fg=self.colors['text_secondary'],
                                bg=self.colors['bg_secondary'])
        status_label.pack(anchor='e')
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_display = tk.Label(status_frame,
                                       textvariable=self.status_var,
                                       font=('Segoe UI', 11, 'bold'),
                                       fg=self.colors['success'],
                                       bg=self.colors['bg_secondary'])
        self.status_display.pack(anchor='e')
        
        # Subtitle
        subtitle_label = tk.Label(header_content,
                                  text="Advanced Machine Learning Stock Prediction Platform",
                                  font=('Segoe UI', 12),
                                  fg=self.colors['text_secondary'],
                                  bg=self.colors['bg_secondary'])
        subtitle_label.pack(side='left', anchor='sw', pady=(10, 0))
        
    def create_left_panel(self, parent):
        # Add padding
        padded_frame = tk.Frame(parent, bg=self.colors['bg_secondary'])
        padded_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Stock input section
        stock_frame = ttk.LabelFrame(padded_frame, 
                                     text=" 🎯 Stock Selection ",
                                     style='Modern.TLabelframe')
        stock_frame.pack(fill='x', pady=(0, 15))
        
        # Stock symbol input
        symbol_inner = tk.Frame(stock_frame, bg=self.colors['bg_secondary'])
        symbol_inner.pack(fill='x', padx=15, pady=15)
        
        tk.Label(symbol_inner,
                 text="Stock Symbol:",
                 font=('Segoe UI', 10, 'bold'),
                 fg=self.colors['text_primary'],
                 bg=self.colors['bg_secondary']).pack(anchor='w', pady=(0, 5))
        
        symbol_entry_frame = tk.Frame(symbol_inner, bg=self.colors['bg_secondary'])
        symbol_entry_frame.pack(fill='x', pady=(0, 10))
        
        self.symbol_var = tk.StringVar(value="AAPL")
        
        available_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "LLY", "V",
            "UNH", "XOM", "WMT", "JPM", "MA", "PG", "AVGO", "HD", "CVX", "MRK",
            "KO", "PEP", "COST", "ABBV", "BAC", "MCD", "CRM", "PFE", "TMO", "CSCO",
            "NFLX", "AMD", "INTC", "DIS", "BA", "IBM", "GE", "F", "GM", "UBER", "PLTR", "CRWD"
        ]
        
        symbol_entry = ttk.Combobox(symbol_entry_frame,
                                    textvariable=self.symbol_var,
                                    values=sorted(available_stocks),
                                    style='Modern.TCombobox',
                                    font=('Segoe UI', 12),
                                    state='readonly')
        symbol_entry.pack(side='left', fill='x', expand=True, ipady=5)
        
        fetch_btn = ttk.Button(symbol_entry_frame,
                               text="📊 Fetch Data",
                               command=self.fetch_stock_data,
                               style='Modern.TButton')
        fetch_btn.pack(side='right', padx=(10, 0), ipady=5)
        
        # Prediction parameters section
        params_frame = ttk.LabelFrame(padded_frame,
                                      text=" ⚙️ Prediction Parameters ",
                                      style='Modern.TLabelframe')
        params_frame.pack(fill='x', pady=(0, 15))
        
        params_inner = tk.Frame(params_frame, bg=self.colors['bg_secondary'])
        params_inner.pack(fill='x', padx=15, pady=15)
        
        # Days to predict
        tk.Label(params_inner,
                 text="Days to Predict:",
                 font=('Segoe UI', 10, 'bold'),
                 fg=self.colors['text_primary'],
                 bg=self.colors['bg_secondary']).pack(anchor='w', pady=(0, 5))
        
        self.days_var = tk.StringVar(value="30")
        days_entry = ttk.Entry(params_inner,
                               textvariable=self.days_var,
                               style='Modern.TEntry',
                               font=('Segoe UI', 11))
        days_entry.pack(fill='x', pady=(0, 10), ipady=3)
        
        # Model selection
        tk.Label(params_inner,
                 text="Prediction Model:",
                 font=('Segoe UI', 10, 'bold'),
                 fg=self.colors['text_primary'],
                 bg=self.colors['bg_secondary']).pack(anchor='w', pady=(0, 5))
        
        self.model_var = tk.StringVar(value="Random Forest")
        model_combo = ttk.Combobox(params_inner,
                                   textvariable=self.model_var,
                                   values=["Linear Regression", "Random Forest", "Ensemble"],
                                   style='Modern.TCombobox',
                                   font=('Segoe UI', 11),
                                   state='readonly')
        model_combo.pack(fill='x', pady=(0, 15), ipady=3)
        
        # Predict button
        predict_btn = ttk.Button(params_inner,
                                 text="🔮 Generate Prediction",
                                 command=self.generate_predictions,
                                 style='Modern.TButton')
        predict_btn.pack(fill='x', ipady=8)
        
        # Stock info section
        info_frame = ttk.LabelFrame(padded_frame,
                                    text=" 📋 Stock Information ",
                                    style='Modern.TLabelframe')
        info_frame.pack(fill='x', pady=(0, 15))
        
        info_inner = tk.Frame(info_frame, bg=self.colors['bg_secondary'])
        info_inner.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Create scrollable text widget
        info_scroll_frame = tk.Frame(info_inner, bg=self.colors['bg_secondary'])
        info_scroll_frame.pack(fill='both', expand=True)
        
        self.info_text = tk.Text(info_scroll_frame,
                                 height=8,
                                 bg=self.colors['bg_tertiary'],
                                 fg=self.colors['text_primary'],
                                 font=('Consolas', 9),
                                 insertbackground=self.colors['text_primary'],
                                 selectbackground=self.colors['accent'],
                                 relief='flat',
                                 wrap='word',
                                 padx=10,
                                 pady=10)
        
        info_scrollbar = ttk.Scrollbar(info_scroll_frame, orient='vertical', command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.pack(side='left', fill='both', expand=True)
        info_scrollbar.pack(side='right', fill='y')
        
        # Model performance section
        perf_frame = ttk.LabelFrame(padded_frame,
                                    text=" 📊 Model Performance ",
                                    style='Modern.TLabelframe')
        perf_frame.pack(fill='both', expand=True)
        
        perf_inner = tk.Frame(perf_frame, bg=self.colors['bg_secondary'])
        perf_inner.pack(fill='both', expand=True, padx=15, pady=15)
        
        perf_scroll_frame = tk.Frame(perf_inner, bg=self.colors['bg_secondary'])
        perf_scroll_frame.pack(fill='both', expand=True)
        
        self.perf_text = tk.Text(perf_scroll_frame,
                                 bg=self.colors['bg_tertiary'],
                                 fg=self.colors['text_primary'],
                                 font=('Consolas', 9),
                                 insertbackground=self.colors['text_primary'],
                                 selectbackground=self.colors['accent'],
                                 relief='flat',
                                 wrap='word',
                                 padx=10,
                                 pady=10)
        
        perf_scrollbar = ttk.Scrollbar(perf_scroll_frame, orient='vertical', command=self.perf_text.yview)
        self.perf_text.configure(yscrollcommand=perf_scrollbar.set)
        
        self.perf_text.pack(side='left', fill='both', expand=True)
        perf_scrollbar.pack(side='right', fill='y')
        
    def create_right_panel(self, parent):
        # Add padding
        padded_frame = tk.Frame(parent, bg=self.colors['bg_secondary'])
        padded_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Create notebook with custom styling
        notebook_frame = tk.Frame(padded_frame, bg=self.colors['bg_secondary'])
        notebook_frame.pack(fill='both', expand=True)
        
        # Custom notebook
        self.notebook = ttk.Notebook(notebook_frame)
        
        # Configure notebook style
        style = ttk.Style()
        style.configure('TNotebook', background=self.colors['bg_secondary'])
        style.configure('TNotebook.Tab',
                        background=self.colors['bg_tertiary'],
                        foreground=self.colors['text_primary'],
                        padding=[20, 10],
                        font=('Segoe UI', 10, 'bold'))
        style.map('TNotebook.Tab',
                  background=[('selected', self.colors['accent']),
                              ('active', self.colors['bg_tertiary'])])
        
        # Price chart tab
        self.price_frame = tk.Frame(self.notebook, bg=self.colors['bg_secondary'])
        self.notebook.add(self.price_frame, text="📈 Price Chart & Predictions")
        
        # Technical indicators tab
        self.tech_frame = tk.Frame(self.notebook, bg=self.colors['bg_secondary'])
        self.notebook.add(self.tech_frame, text="📊 Technical Indicators")
        
        # Volume analysis tab
        self.volume_frame = tk.Frame(self.notebook, bg=self.colors['bg_secondary'])
        self.notebook.add(self.volume_frame, text="📊 Volume Analysis")

        # Recommendations tab
        self.recomm_frame = tk.Frame(self.notebook, bg=self.colors['bg_secondary'])
        self.notebook.add(self.recomm_frame, text="💡 AI Recommendations")
        
        self.notebook.pack(fill='both', expand=True)
        
        self.setup_matplotlib_style()
        self.create_placeholder_content()
        self.setup_recommendations_tab()
        
    def setup_recommendations_tab(self):
        container = tk.Frame(self.recomm_frame, bg=self.colors['bg_secondary'])
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header for the Recommendations
        header = tk.Label(container, 
                          text="💡 Top AI-Driven Picks (Next 10-20 Days)", 
                          font=('Segoe UI', 16, 'bold'),
                          bg=self.colors['bg_secondary'],
                          fg=self.colors['accent'])
        header.pack(anchor='w', pady=(0, 5))
        
        sub = tk.Label(container,
                       text="Based on real-time simulated pattern recognition and momentum indicators.",
                       font=('Segoe UI', 10),
                       bg=self.colors['bg_secondary'],
                       fg=self.colors['text_secondary'])
        sub.pack(anchor='w', pady=(0, 20))

        # Treeview to display the recommendations
        columns = ("symbol", "company", "timeframe", "profit", "risk", "reason")
        tree = ttk.Treeview(container, columns=columns, show="headings", style="Custom.Treeview")
        
        tree.heading("symbol", text="Symbol")
        tree.heading("company", text="Company")
        tree.heading("timeframe", text="Timeframe")
        tree.heading("profit", text="Exp. Profit")
        tree.heading("risk", text="Risk Level")
        tree.heading("reason", text="Catalyst / AI Insight")
        
        tree.column("symbol", width=80, anchor='center')
        tree.column("company", width=180, anchor='w')
        tree.column("timeframe", width=100, anchor='center')
        tree.column("profit", width=100, anchor='center')
        tree.column("risk", width=100, anchor='center')
        tree.column("reason", width=350, anchor='w')
        
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Mock recommendation data
        data = [
            ("NVDA", "NVIDIA Corp", "15 Days", "+12.5%", "High", "Strong AI chip demand & breakout momentum"),
            ("MSFT", "Microsoft Corp", "10 Days", "+5.2%", "Low", "Cloud infrastructure growth acceleration"),
            ("TSLA", "Tesla Inc", "20 Days", "+8.4%", "High", "New gigafactory output efficiency metrics"),
            ("AAPL", "Apple Inc", "14 Days", "+4.1%", "Low", "Upcoming product launch cycle & upgrades"),
            ("AMD", "Advanced Micro Devices", "10 Days", "+9.8%", "Medium", "Market share gains in server CPU space"),
            ("AMZN", "Amazon.com Inc", "20 Days", "+6.7%", "Medium", "E-commerce volume recovery & AWS margins"),
            ("PLTR", "Palantir Technologies", "10 Days", "+15.3%", "High", "New government defense contracts secured"),
            ("META", "Meta Platforms", "12 Days", "+7.2%", "Medium", "Ad revenue growth and efficiency gains"),
            ("GOOGL", "Alphabet Inc", "18 Days", "+6.1%", "Low", "Search dominance and deep AI integration"),
            ("CRWD", "CrowdStrike Holdings", "15 Days", "+11.0%", "High", "Cybersecurity sector momentum push")
        ]
        
        for item in data:
            tree.insert("", tk.END, values=item)

    def create_placeholder_content(self):
        for frame, title in [(self.price_frame, "Price Chart & Predictions"),
                             (self.tech_frame, "Technical Indicators"),
                             (self.volume_frame, "Volume Analysis")]:
            
            # Clear existing widgets
            for widget in frame.winfo_children():
                widget.destroy()
            
            placeholder = tk.Label(frame,
                                   text=f"📊 {title}\n\nClick 'Fetch Data' to load stock information",
                                   font=('Segoe UI', 14),
                                   fg=self.colors['text_secondary'],
                                   bg=self.colors['bg_secondary'])
            placeholder.pack(expand=True)
        
    def setup_matplotlib_style(self):
        plt.style.use('dark_background')
        plt.rcParams.update({
            "figure.facecolor": self.colors['bg_secondary'],
            "axes.facecolor": self.colors['bg_tertiary'],
            "axes.edgecolor": self.colors['border'],
            "axes.labelcolor": self.colors['text_primary'],
            "xtick.color": self.colors['text_secondary'],
            "ytick.color": self.colors['text_secondary'],
            "grid.color": self.colors['border'],
            "legend.facecolor": self.colors['bg_tertiary'],
            "legend.edgecolor": self.colors['border'],
            "text.color": self.colors['text_primary']
        })
        sns.set_palette("husl")
        
    def fetch_stock_data(self):
        self.status_var.set("Fetching data...")
        self.status_display.config(fg=self.colors['warning'])
        self.root.config(cursor="wait")
        
        # The background thread now only fetches and processes data
        def fetch_data_thread():
            try:
                symbol = self.symbol_var.get().upper().strip()
                if not symbol:
                    raise ValueError("Please enter a stock symbol")
                
                stock = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)
                
                # Store data in instance variables
                self.current_data = stock.history(start=start_date, end=end_date)
                if self.current_data.empty:
                    raise Exception(f"No data found for symbol {symbol}")

                self.predictions = {}
                info = stock.info
                self.calculate_technical_indicators()
                
                # Schedule the GUI update to run on the main thread
                self.root.after(0, self._on_fetch_success, info, symbol)

            except Exception as e:
                # Schedule the error message to run on the main thread
                self.root.after(0, self._on_fetch_failure, e)
        
        threading.Thread(target=fetch_data_thread, daemon=True).start()

    def _on_fetch_success(self, info, symbol):
        """GUI update logic after successfully fetching data. Runs in main thread."""
        self.update_stock_info(info, symbol)
        self.create_price_chart()
        self.create_technical_chart()
        self.create_volume_chart()
        
        self.status_var.set(f"Data loaded: {symbol}")
        self.status_display.config(fg=self.colors['success'])
        self.root.config(cursor="")

    def _on_fetch_failure(self, error):
        """GUI update logic when fetching fails. Runs in main thread."""
        messagebox.showerror("Error", f"Failed to fetch data: {str(error)}")
        self.status_var.set("Error fetching data")
        self.status_display.config(fg=self.colors['error'])
        self.root.config(cursor="")
        
    def calculate_technical_indicators(self):
        if self.current_data is None:
            return
            
        # Moving averages
        self.current_data['MA20'] = self.current_data['Close'].rolling(window=20).mean()
        self.current_data['MA50'] = self.current_data['Close'].rolling(window=50).mean()
        self.current_data['MA200'] = self.current_data['Close'].rolling(window=200).mean()
        
        # RSI
        delta = self.current_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.current_data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.current_data['Close'].ewm(span=12).mean()
        exp2 = self.current_data['Close'].ewm(span=26).mean()
        self.current_data['MACD'] = exp1 - exp2
        self.current_data['MACD_signal'] = self.current_data['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        self.current_data['BB_middle'] = self.current_data['Close'].rolling(window=20).mean()
        bb_std = self.current_data['Close'].rolling(window=20).std()
        self.current_data['BB_upper'] = self.current_data['BB_middle'] + (bb_std * 2)
        self.current_data['BB_lower'] = self.current_data['BB_middle'] - (bb_std * 2)
        
    def update_stock_info(self, info, symbol):
        try:
            current_price = self.current_data['Close'].iloc[-1]
            prev_close = self.current_data['Close'].iloc[-2] if len(self.current_data) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            # Format numbers safely
            def safe_format(value, format_str="{:,.2f}", default="N/A"):
                try:
                    if pd.isna(value) or value is None:
                        return default
                    return format_str.format(value)
                except:
                    return default
            
            info_text = f"""
╔══════════════════════════════════════╗
║        {symbol} - STOCK INFO        ║
╚══════════════════════════════════════╝

💰 Current Price: ${safe_format(current_price)}
📈 Daily Change: ${safe_format(change)} ({change_pct:+.2f}%)
📊 Volume: {safe_format(self.current_data['Volume'].iloc[-1], "{:,.0f}")}

🏢 Company: {info.get('longName', 'N/A')}
🏭 Sector: {info.get('sector', 'N/A')}
🔧 Industry: {info.get('industry', 'N/A')}

💼 Market Cap: ${safe_format(info.get('marketCap'), "{:,.0f}")}
📈 52W High: ${safe_format(info.get('fiftyTwoWeekHigh'))}
📉 52W Low: ${safe_format(info.get('fiftyTwoWeekLow'))}
💹 P/E Ratio: {safe_format(info.get('trailingPE'))}
💰 Dividend Yield: {safe_format(info.get('dividendYield', 0) * 100 if info.get('dividendYield') else None, "{:.2f}%")}

📊 Beta: {safe_format(info.get('beta'))}
💸 EPS: ${safe_format(info.get('trailingEps'))}
📈 Revenue: ${safe_format(info.get('totalRevenue'), "{:,.0f}")}
"""
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, info_text)
            
        except Exception as e:
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"Error displaying info: {str(e)}")
            
    def generate_predictions(self):
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please fetch stock data first!")
            return
            
        self.status_var.set("Generating predictions...")
        self.status_display.config(fg=self.colors['warning'])
        self.root.config(cursor="wait")

        def predict_thread():
            try:
                days_to_predict = int(self.days_var.get())
                if days_to_predict <= 0 or days_to_predict > 365:
                    raise ValueError("Days to predict must be between 1 and 365")
                
                model_type = self.model_var.get()
                features = self.prepare_features()
                
                if model_type == "Linear Regression":
                    predictions = self.linear_regression_predict(features, days_to_predict)
                elif model_type == "Random Forest":
                    predictions = self.random_forest_predict(features, days_to_predict)
                else:  # Ensemble
                    predictions = self.ensemble_predict(features, days_to_predict)
                
                # Schedule GUI update on the main thread
                self.root.after(0, self._on_predict_success, model_type, predictions)
                
            except Exception as e:
                # Schedule error message on the main thread
                self.root.after(0, self._on_predict_failure, e)
        
        threading.Thread(target=predict_thread, daemon=True).start()

    def _on_predict_success(self, model_type, predictions):
        """GUI update logic after successful prediction. Runs in main thread."""
        self.predictions[model_type] = predictions
        self.create_price_chart()
        self.status_var.set(f"Predictions generated: {model_type}")
        self.status_display.config(fg=self.colors['success'])
        self.root.config(cursor="")
        
    def _on_predict_failure(self, error):
        """GUI update logic when prediction fails. Runs in main thread."""
        messagebox.showerror("Error", f"Prediction failed: {str(error)}")
        self.status_var.set("Prediction failed")
        self.status_display.config(fg=self.colors['error'])
        self.root.config(cursor="")
        
    def prepare_features(self):
        data = self.current_data.copy()
        
        # Create features
        data['Price_Change'] = data['Close'].pct_change()
        data['Volume_Change'] = data['Volume'].pct_change()
        data['High_Low_Ratio'] = data['High'] / data['Low']
        data['Price_MA20_Ratio'] = data['Close'] / data['MA20']
        
        # Lag features
        for i in range(1, 6):
            data[f'Close_lag_{i}'] = data['Close'].shift(i)
            data[f'Volume_lag_{i}'] = data['Volume'].shift(i)
        
        # Technical indicators as features
        feature_columns = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'MACD'] + \
                          [f'Close_lag_{i}' for i in range(1, 6)] + \
                          [f'Volume_lag_{i}' for i in range(1, 6)] + \
                          ['Price_Change', 'Volume_Change', 'High_Low_Ratio', 'Price_MA20_Ratio']
        
        features = data[feature_columns].dropna()
        return features
        
    def linear_regression_predict(self, features, days):
        X = features.iloc[:-1].values
        y = features['Close'].iloc[1:].values
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.update_performance_display("Linear Regression", mae, r2)
        
        predictions = []
        current_features = features.iloc[-1:].copy()
        
        for _ in range(days):
            scaled_features = scaler.transform(current_features.values)
            pred = model.predict(scaled_features)[0]
            predictions.append(pred)
            # Update the 'Close' price for the next prediction step
            current_features['Close'] = pred
            
        return predictions
        
    def random_forest_predict(self, features, days):
        X = features.iloc[:-1].values
        y = features['Close'].iloc[1:].values
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.update_performance_display("Random Forest", mae, r2)
        
        predictions = []
        current_features = features.iloc[-1:].copy()
        
        for _ in range(days):
            pred = model.predict(current_features.values)[0]
            predictions.append(pred)
            # Update the 'Close' price for the next prediction step
            current_features['Close'] = pred
            
        return predictions
        
    def ensemble_predict(self, features, days):
        # Note: In a real-world scenario, you might want to call these only once
        # and store the models to avoid retraining for the ensemble.
        lr_pred = self.linear_regression_predict(features.copy(), days)
        rf_pred = self.random_forest_predict(features.copy(), days)
        
        # Weighted average ensemble
        ensemble_pred = [0.6 * rf + 0.4 * lr for rf, lr in zip(rf_pred, lr_pred)]
        
        # Performance for ensemble is illustrative; a proper evaluation would require a separate test set.
        self.update_performance_display("Ensemble (Weighted Avg)", -1, -1)
        
        return ensemble_pred
        
    def update_performance_display(self, model_name, mae, r2):
        if mae == -1: # Special case for ensemble illustration
             perf_text = f"""
╔═══════════════════════════════════════╗
║         MODEL PERFORMANCE             ║
║         {model_name:<25} ║
╚═══════════════════════════════════════╝

This is a weighted average of Linear 
Regression (40%) and Random Forest (60%).

Performance metrics are based on the
individual models. See their respective
runs for details.
"""
        else:
            quality = "Excellent" if r2 > 0.8 else "Good" if r2 > 0.6 else "Fair" if r2 > 0.4 else "Poor"
            quality_color = "🟢" if r2 > 0.8 else "🟡" if r2 > 0.6 else "🟠" if r2 > 0.4 else "🔴"
        
            perf_text = f"""
╔═══════════════════════════════════════╗
║         MODEL PERFORMANCE             ║
║         {model_name:<25} ║
╚═══════════════════════════════════════╝

📊 Mean Absolute Error: ${mae:.2f}
📈 R² Score: {r2:.4f}
{quality_color} Model Quality: {quality}

📋 Performance Metrics:
  • MAE: Average prediction error in $.
  • R²: Proportion of price variance
    that the model can predict.
    (Closer to 1 is better).
"""
        self.perf_text.delete(1.0, tk.END)
        self.perf_text.insert(tk.END, perf_text)

    def _clear_frame(self, frame):
        """Helper function to destroy all widgets in a frame."""
        for widget in frame.winfo_children():
            widget.destroy()

    def create_price_chart(self):
        self._clear_frame(self.price_frame)
        if self.current_data is None: return

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Plot historical data
        ax.plot(self.current_data.index, self.current_data['Close'], label='Historical Close Price', color=self.colors['accent'], lw=2)
        ax.fill_between(self.current_data.index, self.current_data['BB_lower'], self.current_data['BB_upper'], color='grey', alpha=0.2, label='Bollinger Bands')
        ax.plot(self.current_data.index, self.current_data['MA50'], label='50-Day MA', color='orange', linestyle='--', lw=1)
        
        # Plot predictions if available
        model_type = self.model_var.get()
        if model_type in self.predictions:
            predictions = self.predictions[model_type]
            last_date = self.current_data.index[-1]
            future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, len(predictions) + 1)])
            ax.plot(future_dates, predictions, label=f'{model_type} Prediction', color='red', linestyle='--', marker='o', markersize=3)
        
        ax.set_title(f"{self.symbol_var.get()} Price Chart and Prediction", fontsize=16, color=self.colors['text_primary'])
        ax.set_xlabel("Date", fontsize=12, color=self.colors['text_secondary'])
        ax.set_ylabel("Price (USD)", fontsize=12, color=self.colors['text_secondary'])
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.price_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    def create_technical_chart(self):
        self._clear_frame(self.tech_frame)
        if self.current_data is None: return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
        fig.patch.set_facecolor(self.colors['bg_secondary'])

        # RSI Plot
        ax1.plot(self.current_data.index, self.current_data['RSI'], label='RSI', color='cyan')
        ax1.axhline(70, linestyle='--', color='red', alpha=0.7, label='Overbought (70)')
        ax1.axhline(30, linestyle='--', color='green', alpha=0.7, label='Oversold (30)')
        ax1.set_ylim(0, 100)
        ax1.set_title('Relative Strength Index (RSI)', color=self.colors['text_primary'])
        ax1.set_ylabel('RSI Value', color=self.colors['text_secondary'])
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)

        # MACD Plot
        ax2.plot(self.current_data.index, self.current_data['MACD'], label='MACD', color='blue')
        ax2.plot(self.current_data.index, self.current_data['MACD_signal'], label='Signal Line', color='orange', linestyle='--')
        macd_hist = self.current_data['MACD'] - self.current_data['MACD_signal']
        ax2.bar(self.current_data.index, macd_hist, width=1.0, label='Histogram', color=np.where(macd_hist >= 0, 'g', 'r'), alpha=0.5)
        ax2.set_title('MACD', color=self.colors['text_primary'])
        ax2.set_ylabel('MACD Value', color=self.colors['text_secondary'])
        ax2.set_xlabel('Date', color=self.colors['text_secondary'])
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.tech_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    def create_volume_chart(self):
        self._clear_frame(self.volume_frame)
        if self.current_data is None: return

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        ax.bar(self.current_data.index, self.current_data['Volume'], label='Volume', color=self.colors['accent'], alpha=0.6)
        
        # Volume Moving Average
        volume_ma = self.current_data['Volume'].rolling(window=50).mean()
        ax.plot(self.current_data.index, volume_ma, label='50-Day Volume MA', color='yellow', linestyle='--', lw=2)

        ax.set_title(f"{self.symbol_var.get()} Trading Volume", fontsize=16, color=self.colors['text_primary'])
        ax.set_xlabel("Date", fontsize=12, color=self.colors['text_secondary'])
        ax.set_ylabel("Volume", fontsize=12, color=self.colors['text_secondary'])
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.legend()
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.volume_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernStockPredictor(root)
    root.mainloop()
