# MarketMind Graph 🧠

An intelligent market analysis platform that leverages knowledge graphs and AI to provide deep insights into the Magnificent Seven tech companies (Apple, Microsoft, Alphabet, Amazon, NVIDIA, Meta, and Tesla).

## Features 🌟

- **Real-time News Analysis**: Automatically fetches and processes news articles for major tech companies
- **Knowledge Graph Integration**: Builds and maintains a dynamic knowledge graph of market relationships
- **AI-Powered Insights**: Uses Mistral Large 2 model for advanced natural language processing and analysis
- **Semantic Search**: Find relevant information using advanced semantic similarity search
- **Interactive UI**: Clean, modern interface built with Streamlit
- **Pro Search Mode**: Enhanced query processing with automatic question refinement
- **Trending Questions**: Track and display popular user queries
- **Content Generation**: Automatically generate consolidated reports from chat history

## Architecture 🏗️

The application consists of three main components:

1. **Data Collection (`snowflake_sync.py`)**
   - Fetches news articles using yfinance API
   - Processes and cleans article content
   - Stores data in Snowflake database
   - Runs on a 6-hour schedule

2. **Knowledge Graph Processing (`snowflake_operations.py`)**
   - Manages all Snowflake database operations
   - Handles entity extraction and relationship mapping
   - Implements semantic search functionality
   - Manages query caching and cleanup

3. **Web Interface (`app.py`)**
   - Streamlit-based user interface
   - Real-time chat functionality
   - Dynamic content rendering
   - Article preview modal system

## Technologies Used 🛠️

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: Snowflake
- **AI Models**: 
  - Mistral Large 2 (via Snowflake Cortex)
  - Snowflake Arctic Embedding Model
- **APIs**: yfinance
- **Key Libraries**:
  - `snowflake-connector-python`
  - `beautifulsoup4`
  - `pandas`
  - `schedule`

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/marketmind-graph.git
cd marketmind-graph
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up Snowflake credentials:
Create a `.streamlit/secrets.toml` file with:
```toml
[snowflake]
user = "your_username"
password = "your_password"
account = "your_account"
warehouse = "your_warehouse"
database = "your_database"
schema = "your_schema"
```

4. Initialize the database:
```bash
python init_db.py
```

## Usage 💡

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Run the news sync service:
```bash
python snowflake_sync.py --schedule
```

## Key Features in Detail 📝

### News Synchronization
- Automated news fetching every 6 hours
- Intelligent deduplication
- Content extraction and cleaning
- Error handling and retry mechanisms

### Knowledge Graph
- Entity extraction with contextual awareness
- Relationship mapping with confidence scoring
- Semantic embedding for enhanced search
- Incremental graph building

### Query Processing
- Smart query refinement
- Multi-query parallel processing
- Context-aware response generation
- Citation and reference management

### Content Generation
- Automatic report generation
- Reference preservation
- Entity relationship visualization
- Downloadable markdown format

## Project Structure 📁

```
marketmind-graph/
├── app.py                 # Main Streamlit application
├── snowflake_sync.py      # News synchronization service
├── snowflake_operations.py # Database operations
├── util.py                # Utility functions
├── requirements.txt       # Project dependencies
├── .streamlit/           # Streamlit configuration
│   └── secrets.toml      # Credentials (git-ignored)
└── logs/                 # Application logs
```

## Contributing 🤝

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Error Handling and Logging 🔍

The application implements comprehensive error handling and logging:
- Detailed logging of all operations
- Automatic retry mechanisms for API calls
- Transaction management for database operations
- Error notifications in the UI

## Performance Considerations 🚀

- Query caching system
- Parallel query processing
- Incremental knowledge graph updates
- Optimized database operations
- Resource-efficient data synchronization

## Security 🔒

- Secure credential management
- API rate limiting
- Input validation and sanitization
- Secure database connections
- No sensitive data exposure


## Acknowledgments 🙏

- Snowflake for providing the database and AI infrastructure
- Streamlit for the fantastic web framework
- Yahoo Finance for financial data access
- The open source community for various tools and libraries used in this project



---

Made with ❤️ by Angu Krishna