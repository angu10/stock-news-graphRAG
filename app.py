import streamlit as st
import streamlit.components.v1 as html
from datetime import datetime
import json
from typing import List, Dict, Tuple
from snowflake_operations import SnowflakeOperations
import pandas as pd
from util import json_cleanup
import asyncio

# Initialize Snowflake Operations
snowflake_ops = SnowflakeOperations()

# Initialize session states (removed article viewer state)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "pro_search" not in st.session_state:
    st.session_state.pro_search = False
if "current_question" not in st.session_state:
    st.session_state.current_question = None

# Run cache cleanup daily
if "last_cleanup" not in st.session_state:
    st.session_state.last_cleanup = datetime.now()
    snowflake_ops.cleanup_cache()
elif (datetime.now() - st.session_state.last_cleanup).days >= 1:
    snowflake_ops.cleanup_cache()
    st.session_state.last_cleanup = datetime.now()

st.set_page_config(
    page_title="MarketMind Graph",
    page_icon="🧠",  
    layout="centered",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<script>
    function showModal(modalId) {
        var modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = "block";
            // Prevent background scrolling
            document.body.style.overflow = 'hidden';
        }
    }
    
    function closeModal(modalId) {
        var modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = "none";
            // Restore background scrolling
            document.body.style.overflow = 'auto';
        }
    }
    
    // Close modal when clicking outside
    window.onclick = function(event) {
        if (event.target.classList.contains('modal')) {
            closeModal(event.target.id);
        }
    }
</script>""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
    .stApp {
            background-color: #1a1a1a;
            color: #e6e6e6;
    }
    .modal {
        display: none;
        position: fixed;
        z-index: 1000;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.7);
        overflow-y: auto;
    }
    
    .modal-content {
        background-color: #2d2d2d;
        margin: 5% auto;  /* Adjusted margin */
        padding: 20px;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        width: 80%;
        max-height: 80vh;  /* Increased height */
        overflow-y: auto;
        position: relative;  /* Added position */
    }
    .modal-header {
        padding-bottom: 10px;
        border-bottom: 1px solid #4a4a4a;
        margin-bottom: 15px;
    }
    
    .modal-body {
        padding: 15px 0;
    }
    
    .modal-footer {
        padding-top: 10px;
        border-top: 1px solid #4a4a4a;
        margin-top: 15px;
    }
    
    .close-button {
        color: #aaa;
        float: right;
        font-size: 28px;
        font-weight: bold;
        cursor: pointer;
        position: sticky;  /* Made sticky */
        top: 10px;
        right: 10px;
    }
    
    .close-button:hover {
        color: #fff;
    }
    
    .reference-link {
        text-decoration: none;
        color: #3b82f6;
        cursor: pointer;
    }
    
    .reference-link:hover {
        text-decoration: underline;
    }
    
    .reference-metrics {
        display: flex;
        gap: 10px;
        margin-top: 8px;
    }
    
    .metric-badge {
        background-color: #374151;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        color: #9ca3af;
    }
    .trending-question {
    padding: 8px 12px;
    background-color: rgba(59, 130, 246, 0.1);
    border-radius: 6px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.2s;
    }
    .trending-question:hover {
        background-color: rgba(59, 130, 246, 0.2);
    }
    .trending-meta {
        font-size: 0.8em;
        color: #666;
    }
    .sentiment-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .sentiment-positive {
        background-color: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    .sentiment-negative {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    .sentiment-neutral {
        background-color: rgba(234, 179, 8, 0.1);
        border: 1px solid rgba(234, 179, 8, 0.2);
    }
    .sentiment-driver {
        background-color: rgba(59, 130, 246, 0.1);
        padding: 4px 8px;
        border-radius: 4px;
        display: inline-block;
        margin: 2px;
    }
</style>
""",
    unsafe_allow_html=True,
)


def generate_follow_up_questions(context: str, current_query: str) -> List[str]:
    """Generate relevant follow-up questions based on the context and current query"""
    try:
        with snowflake_ops._get_connection() as conn:
            cursor = conn.cursor()

            prompt = f"""
            Based on this context and current question:
            
            Current Question: {current_query}
            Context: {context}
            
            Generate 3 relevant follow-up questions that would help users dive deeper into:
            1. Related market implications
            2. Historical context or trends
            3. Connected entities or events
            4. Potential future impacts
            
            Return only the questions in a JSON array format.
            Make each question specific and directly related to the context.
            """

            cursor.execute(
                """
                SELECT SNOWFLAKE.CORTEX.COMPLETE(
                    'mistral-large2',
                    %s
                )
            """,
                (prompt,),
            )

            result = json_cleanup(cursor.fetchone()[0])
            follow_up_questions = json.loads(result)
            return follow_up_questions[:5]  # Limit to 5 questions

    except Exception as e:
        st.error(f"Error generating follow-up questions: {str(e)}")
        return []


def handle_suggested_question(question: str):
    """Handle when a suggested question is clicked"""
    st.session_state.current_question = question
    st.session_state.messages.append({"role": "user", "content": question})
    st.rerun()


def process_query_with_graph(query: str, context: Dict,is_original_query: bool = False) -> Dict:
    """Process query using knowledge graph data and news context with inline references"""
    try:
        with snowflake_ops._get_connection() as conn:
            cursor = conn.cursor()

            # First, classify the question type
            question_type = "GENERAL"
            if is_original_query:
                #st.write(f"is_original_query {is_original_query}")
                question_type = snowflake_ops.classify_question_type(query)

            # Extract entities and relationships from the query
            graph_data = snowflake_ops.query_graph(query)

            # Base prompt for both types
            base_prompt = f"""
            Analyze this question using both news articles and knowledge graph data:
            
            Question: {query}
            News Context: {context.get('news_context', '')}
            Graph Analysis: {graph_data}
            """
            if question_type == "SENTIMENT":
                # For sentiment questions, add sentiment-specific instructions
                prompt = base_prompt + """
                Provide a comprehensive answer that focuses on market sentiment and includes:
                1. Overall market mood and sentiment
                2. Key factors driving the sentiment
                3. Recent sentiment shifts or trends
                4. Market implications
                5. Use inline citations in format [1], [2], etc.
                
                IMPORTANT: 
                1. For EVERY piece of information from a source, include an inline citation like [1], [2], etc.
                2. For each reference, provide both a relevance_score and a confidence_score
                """
            else:
                # For general questions, use standard instructions
                prompt = base_prompt + """
            
                Provide a comprehensive answer that:
                1. Addresses the main question
                2. Incorporates relevant entity relationships
                3. Uses inline citations in format [1], [2], etc. when referencing specific information
                4. Highlights key insights
                5. Includes market sentiment
                
                IMPORTANT: 
                1. For EVERY piece of information from a source, include an inline citation like [1], [2], etc.
                2.For each reference, provide both a relevance_score (how relevant the source is to the query)
                and a confidence_score (how confident you are in the source's information)
                """
            prompt += """
                Format the response in this JSON structure:
                {{
                    "answer": "detailed_answer_with_[1]_style_inline_citations",
                    "key_entities": [
                        {{
                            "name": "entity_name",
                            "type": "entity_type",
                            "relevance": "explanation_of_relevance"
                        }}
                    ],
                    "references": [
                        {{
                            "title": "article_title",
                            "source": "source_name",
                            "date": "publication_date",
                            "link": "article_link",
                            "relevance_score": score,
                            "confidence_score": score_between_0_and_1,
                        }}
                    ],
                    "confidence_score": confidence_between_0_and_1
                }}
                
                Ensure that every reference in the references list is cited at least once in the answer text using [n] format.
            """

            cursor.execute(
                """
                SELECT SNOWFLAKE.CORTEX.COMPLETE(
                        'mistral-large2',
                        [{
                        'role': 'user',
                        'content': %s}],
                        {
                            'max_tokens': 8192
                        }
                    )
            """,
                (prompt,),
            )

            results = json.loads(json_cleanup(cursor.fetchone()[0]))
            result = json.loads(results["choices"][0]["messages"])

            # For sentiment questions, add sentiment analysis
            if question_type == "SENTIMENT":
                sentiment_result = snowflake_ops.analyze_sentiment(result["answer"])
                result["sentiment_analysis"] = sentiment_result

            # Verify inline citations exist and match references
            references = result.get("references", [])
            answer = result.get("answer", "")

            # Ensure each reference has at least one citation
            for i, ref in enumerate(references, 1):
                if f"[{i}]" not in answer:
                    # If reference isn't cited, append a relevant citation
                    result["answer"] = f"{answer} [{i}]"

            return result

    except Exception as e:
        st.error(f"Error processing query with graph: {str(e)}")
        return {
            "answer": "Error processing your query",
            "key_entities": [],
            "references": [],
            "confidence_score": 0,
        }


async def process_query_with_graph_async(query: str, context: Dict,is_original_query: bool = False) -> Dict:
    """Asynchronous wrapper for process_query_with_graph"""
    # Create new event loop for the async operation
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, process_query_with_graph, query, context,is_original_query)


async def process_queries_parallel(
    queries: List[str], context: Dict, status,original_query: str
) -> List[Dict]:
    """Process multiple queries in parallel"""
    tasks = []
    for i, query in enumerate(queries):
        status.update(label=f"🔄 Processing query {i+1}/{len(queries)}...")
        task = process_query_with_graph_async(query, context, is_original_query=(query == original_query))
        tasks.append(task)

    return await asyncio.gather(*tasks)


async def process_all_queries_async(
    original_query: str, query_analysis: Dict, status
) -> Tuple[Dict, List[Dict]]:
    """Asynchronous version of process_all_queries"""
    context = {"news_context": ""}

    # Combine original query and refined queries
    refined_questions = query_analysis.get("refined_questions", [])
    if len(refined_questions) > 2:
        refined_questions = refined_questions[:2]
    all_queries = [original_query] + refined_questions

    # Process all queries in parallel
    status.update(label="📝 Processing queries in parallel...")
    responses = await process_queries_parallel(all_queries, context, status,original_query)

    # Format responses
    all_responses = [
        {"query": query, "response": response}
        for query, response in zip(all_queries, responses)
    ]

    def calculate_response_score(response):
        confidence = response["response"].get("confidence_score", 0)
        entity_relevance = len(response["response"].get("key_entities", [])) * 0.1
        reference_quality = sum(
            ref.get("relevance_score", 0)
            for ref in response["response"].get("references", [])
        ) / max(len(response["response"].get("references", [])), 1)
        return confidence * 0.5 + entity_relevance * 0.2 + reference_quality * 0.3

    # Select best response
    best_response = max(all_responses, key=calculate_response_score)
    return best_response["response"], all_responses


def process_all_queries(
    original_query: str, query_analysis: Dict
) -> Tuple[Dict, List[Dict]]:
    """Main function that orchestrates async processing"""
    with st.status("Processing query...", expanded=True) as status:
        try:
            # Create new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run async processing
            best_response, all_responses = loop.run_until_complete(
                process_all_queries_async(original_query, query_analysis, status)
            )

            status.update(label="✨ Analysis complete!", state="complete")
            return best_response, all_responses

        except Exception as e:
            status.error(f"Error processing queries: {str(e)}")
            raise
        finally:
            loop.close()


def show_article_content(link: str, ref_idx: str):
    """Fetch and display article content in a modal"""
    try:
        content = snowflake_ops.show_article_contents(link, ref_idx)
        if content:
            modal_id = f"articleModal_{ref_idx}"
            st.markdown(
                f"""
                    <div class="modal" id="{modal_id}">
                        <div class="modal-content">
                            <div class="modal-header">
                                <span class="close-button" onclick="closeModal('{modal_id}')">&times;</span>
                                <h2>Article Preview</h2>
                            </div>
                            <div class="modal-body">
                                {content}
                            </div>
                            <div class="modal-footer">
                                <a href="{link}" target="_blank" style="color: #3b82f6;">Read full article</a>
                            </div>
                        </div>
                    </div>
                    """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"[Read article on source website]({link})")
    except Exception as e:
        st.error(f"Error fetching article content: {str(e)}")


def format_reference(idx: int, ref: Dict):
    """Format a single reference with confidence and relevance scores"""
    confidence_score = ref.get("confidence_score", 0)
    relevance_score = ref.get("relevance_score", 0)
    modal_id = f"articleModal_{idx}"

    return f"""
    <div style="
        background-color: #2d2d2d;
        border-radius: 6px;
        padding: 12px;
        margin: 8px 0;
        border: 1px solid #4a4a4a;
    ">
        <div style="font-size: 0.8em; color: #9ca3af; margin-bottom: 4px;">
            Reference [{idx}]
        </div>
        <div style="font-weight: bold;">
            <a href="{ref['link']}" target="_blank" class="reference-link">
                {ref['title']}
            </a>
        </div>
        <div style="font-size: 0.9em; margin-top: 4px;">
            Source: {ref['source']} • {ref['date']}
        </div>
        <div class="reference-metrics">
            <span class="metric-badge">
                Confidence: {confidence_score:.0%}
            </span>
            <span class="metric-badge">
                Relevance: {relevance_score:.0%}
            </span>
        </div>
    </div>
    """


def display_trending_questions():
    trending = snowflake_ops.get_trending_questions(limit=5)

    if trending:
        st.sidebar.markdown("### 🔥 Trending Questions")

        # Add custom CSS for the trending questions
        st.markdown(
            """
        <style>
        .trending-question {
            padding: 8px 12px;
            background-color: rgba(59, 130, 246, 0.1);
            border-radius: 6px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .trending-question:hover {
            background-color: rgba(59, 130, 246, 0.2);
        }
        .trending-meta {
            font-size: 0.8em;
            color: #666;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
        # print(trending)
        for idx, question in enumerate(trending):
            unique_key = f"trending_{idx}_{hash(question['query'])}"

            if st.sidebar.button(
                f"💬 {question['query']}",
                key=unique_key,
                help=f"Asked {question['frequency']} times",
            ):
                st.session_state.messages.append(
                    {"role": "user", "content": question["query"]}
                )
                # Get cached response or process new query
                cached_response = snowflake_ops.get_cached_response(question["query"])
                
                if cached_response:
                    if isinstance(cached_response, str):
                        best_response = json.loads(cached_response)
                    else:
                        best_response = cached_response
                        
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": best_response["answer"],
                        "references": best_response.get("references", []),
                        "key_entities": best_response.get("key_entities", [])
                    })
                else:
                    # Process query
                    query_analysis = {"refined_questions": []} if not st.session_state.pro_search else snowflake_ops.generate_refined_queries(question["query"])
                    best_response, all_responses = process_all_queries(question["query"], query_analysis)
                    
                    # Generate follow-up questions if pro search is enabled
                    if st.session_state.pro_search:
                        follow_up_questions = generate_follow_up_questions(best_response["answer"], question["query"])
                        st.session_state.suggested_questions = follow_up_questions
                    
                    # Cache the response
                    cache_data = {
                        "answer": best_response["answer"],
                        "references": best_response["references"],
                        "key_entities": best_response.get("key_entities", []),
                    }
                    snowflake_ops.cache_query_response(question["query"], cache_data)
                    
                    # Add response to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": best_response["answer"],
                        "references": best_response["references"],
                        "key_entities": best_response.get("key_entities", [])
                    })
                
                st.rerun()


# Sidebar with dark theme 
st.sidebar.title("🧠 MarketMind Graph")
last_refresh = snowflake_ops.get_last_run(process_name="news_sync")
st.sidebar.write(f"Last Data Refresh: {last_refresh}")

# Main title section with neural network inspired gradient
st.markdown("""
<div style='background: linear-gradient(135deg, #0f172a 0%, #1e40af 50%, #312e81 100%); 
            padding: 25px; 
            border-radius: 12px; 
            margin-bottom: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);'>
    <div style='margin-bottom: 8px;'>
        <h1 style='color: white; font-size: 2.5em; margin-bottom: 10px; font-weight: 600;'>MarketMind Graph</h1>
        <p style='color: #94a3b8; font-size: 1.2em; margin-bottom: 20px;'>Connecting the Dots in Market Intelligence</p>
    </div>
    <div style='background: rgba(255, 255, 255, 0.05); 
                padding: 15px; 
                border-radius: 8px; 
                border: 1px solid rgba(255, 255, 255, 0.1);'>
        <p style='color: #e2e8f0; margin-bottom: 10px;'>Analyze the Magnificent Seven through our intelligent graph network:</p>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px;'>
            <div style='background: rgba(0, 0, 0, 0.2); padding: 8px; border-radius: 6px; text-align: center;'>
                <span style='color: #60a5fa;'>AAPL</span> <span style='color: white;'>Apple</span>
            </div>
            <div style='background: rgba(0, 0, 0, 0.2); padding: 8px; border-radius: 6px; text-align: center;'>
                <span style='color: #60a5fa;'>MSFT</span> <span style='color: white;'>Microsoft</span>
            </div>
            <div style='background: rgba(0, 0, 0, 0.2); padding: 8px; border-radius: 6px; text-align: center;'>
                <span style='color: #60a5fa;'>GOOGL</span> <span style='color: white;'>Alphabet</span>
            </div>
            <div style='background: rgba(0, 0, 0, 0.2); padding: 8px; border-radius: 6px; text-align: center;'>
                <span style='color: #60a5fa;'>AMZN</span> <span style='color: white;'>Amazon</span>
            </div>
            <div style='background: rgba(0, 0, 0, 0.2); padding: 8px; border-radius: 6px; text-align: center;'>
                <span style='color: #60a5fa;'>NVDA</span> <span style='color: white;'>NVIDIA</span>
            </div>
            <div style='background: rgba(0, 0, 0, 0.2); padding: 8px; border-radius: 6px; text-align: center;'>
                <span style='color: #60a5fa;'>META</span> <span style='color: white;'>Meta</span>
            </div>
            <div style='background: rgba(0, 0, 0, 0.2); padding: 8px; border-radius: 6px; text-align: center;'>
                <span style='color: #60a5fa;'>TSLA</span> <span style='color: white;'>Tesla</span>
            </div>
        </div>
    </div>
</div>

<div style='background: rgba(30, 41, 59, 0.5); 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);'>
    <h3 style='color: #e2e8f0; margin-bottom: 10px;'>🔮 AI-Powered Market Analysis</h3>
    <p style='color: #cbd5e1;'>
        Leverage our advanced knowledge graph technology to uncover hidden connections, analyze market trends, 
        and generate comprehensive insights about the world's most influential tech companies.
    </p>
    <div style='display: flex; gap: 20px; margin-top: 15px;'>
        <div style='color: #94a3b8;'>🔍 Deep Analysis</div>
        <div style='color: #94a3b8;'>🔄 Real-time Updates</div>
        <div style='color: #94a3b8;'>📊 Smart Synthesis</div>
    </div>
</div>
""", unsafe_allow_html=True)


st.warning("📅 Data Coverage: Limited to last 30 days  and Magnificent Seven stocks only", icon="⚠️")

# Add a description of what the app does
st.markdown(
    """
Ask questions about news and market trends related to the Magnificent Seven - 
the most influential tech companies driving market growth. Get insights about Apple, 
Microsoft, Alphabet, Amazon, NVIDIA, Meta, and Tesla.
**Note**: 
- News data is limited to the last 30 days
- For other stocks or historical data, please use financial data providers
"""
)


def format_citation(text, citations):
    """Format citations in text with styled badges"""
    for i, citation in enumerate(citations, 1):
        text = text.replace(f"[{i}]", f'<span class="citation-badge">[{i}]</span>')
    return text


# Display chat history with dark theme
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "references" in message:
            # Format the message content with citations
            formatted_content = format_citation(
                message["content"], message["references"]
            )
            st.markdown(formatted_content, unsafe_allow_html=True)

            # Display sentiment analysis if available
            if "sentiment_analysis" in message:
                sentiment = message["sentiment_analysis"]
                sentiment_class = f"sentiment-{sentiment['sentiment'].lower()}"
                
                st.markdown(
                    f"""
                    <div class="sentiment-box {sentiment_class}">
                        <h3>📊 Market Sentiment Analysis</h3>
                        <p><strong>Overall Sentiment:</strong> {sentiment['sentiment'].title()} 
                        (Confidence: {sentiment['confidence']:.0%})</p>
                        
                        <p><strong>Key Sentiment Drivers:</strong></p>
                        <div>
                            {"".join([f'<span class="sentiment-driver">{driver}</span>' for driver in sentiment['sentiment_drivers']])}
                        </div>
                        
                        <p><strong>Market Implications:</strong><br>
                        {sentiment['market_implications']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="references-section">', unsafe_allow_html=True)
            st.markdown("### 📚 References")

            for idx, ref in enumerate(message["references"], 1):
                st.markdown(format_reference(idx, ref), unsafe_allow_html=True)
                show_article_content(ref["link"], str(idx))

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write(message["content"])
        if "key_entities" in message:
            st.write("---")
            st.write("🔑 Key Entities:")
            for entity in message["key_entities"]:
                st.write(
                    f"- **{entity['name']}** ({entity['type']}): {entity['relevance']}"
                )

# Display suggested questions
if st.session_state.suggested_questions:
    st.write("---")
    st.write("❓ Suggested Follow-up Questions:")
    cols = st.columns(2)
    for i, question in enumerate(st.session_state.suggested_questions):
        if cols[i % 2].button(question, key=f"follow_up_{i}"):
            st.session_state.messages.append({"role": "user", "content": question})

            status_placeholder = st.empty()
            status_placeholder.info("Processing follow-up question...")

            if st.session_state.pro_search:
                query_analysis = snowflake_ops.generate_refined_queries(question)
            else:
                query_analysis = {"refined_questions": []}

            best_response, all_responses = process_all_queries(question, query_analysis)

            if st.session_state.pro_search:
                follow_up_questions = generate_follow_up_questions(
                    best_response["answer"], question
                )
                st.session_state.suggested_questions = follow_up_questions

            status_placeholder.empty()

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": best_response["answer"],
                    "references": best_response["references"],
                    "key_entities": best_response.get("key_entities", []),
                }
            )
            st.rerun()

# Chat input container with pro search toggle
chat_container = st.container()
col1, col2 = chat_container.columns([4, 1])

# Pro search toggle
with col2:
    pro_search_value = st.toggle("Pro Search 🔍", st.session_state.pro_search)
    if pro_search_value != st.session_state.pro_search:
        st.session_state.pro_search = pro_search_value
        st.session_state.suggested_questions = []
        st.rerun()

    # Chat input
    with col1:
        user_input = st.chat_input("Ask about stock news...")


# Process user input (keeping all existing processing logic)
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    status_placeholder = st.empty()
    cached_response = snowflake_ops.get_cached_response(user_input)

    if cached_response:
        best_response = cached_response
        if isinstance(cached_response, str):
            best_response = json.loads(cached_response)
        status_placeholder.info("Retrieved from cache...")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": best_response["answer"],
                "references": best_response.get("references", []),
                "key_entities": best_response.get("key_entities", []),
            }
        )
    else:
        # Process based on pro search setting
        if st.session_state.pro_search:
            status_placeholder.info("Processing your Pro-Search query...")
            query_analysis = snowflake_ops.generate_refined_queries(user_input)

            if query_analysis.get("breakdown"):
                analysis_text = "Query Analysis:\n" + "\n".join(
                    [f"- {step}" for step in query_analysis["breakdown"]]
                )
                st.write(analysis_text)

            best_response, all_responses = process_all_queries(
                user_input, query_analysis
            )

            follow_up_questions = generate_follow_up_questions(
                best_response["answer"], user_input
            )
            st.session_state.suggested_questions = follow_up_questions

            cache_data = {
                "answer": best_response["answer"],
                "references": best_response["references"],
                "key_entities": best_response.get("key_entities", []),
                "analysis": {
                    "breakdown": query_analysis.get("breakdown", []),
                    "all_responses": all_responses,
                },
            }
            snowflake_ops.cache_query_response(user_input, cache_data)
        else:
            status_placeholder.info("Processing your query...")
            query_analysis = {"refined_questions": []}
            best_response, all_responses = process_all_queries(
                user_input, query_analysis
            )
            cache_data = {
                "answer": best_response["answer"],
                "references": best_response["references"],
                "key_entities": best_response.get("key_entities", []),
            }
            snowflake_ops.cache_query_response(user_input, cache_data)

        status_placeholder.empty()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": best_response["answer"],
                "references": best_response["references"],
                "key_entities": best_response.get("key_entities", []),
            }
        )

        if st.session_state.pro_search:
            with st.expander("See detailed analysis"):
                st.write("Query Analysis Steps:")
                for step in query_analysis.get("breakdown", []):
                    st.write(f"- {step}")

                st.write("\nAll Responses Analyzed:")
                for response in all_responses:
                    st.write(f"\nQuery: {response['query']}")
                    st.write(
                        f"Confidence Score: {response['response'].get('confidence_score', 0)}"
                    )
                    st.write(
                        "Key Entities:",
                        len(response["response"].get("key_entities", [])),
                    )
                    st.write("---")

    st.rerun()


display_trending_questions()

st.sidebar.markdown("---")
st.sidebar.markdown("## 📝 Content Generation")


if st.sidebar.button("Generate Consolidated Content"):
    if (
        len(
            [msg for msg in st.session_state.messages if msg.get("role") == "assistant"]
        )
        > 0
    ):
        with st.spinner("Generating consolidated content..."):
            content = snowflake_ops.generate_consolidated_contents(
                st.session_state.messages
            )
            if content:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"# {content['title']}")
                with col2:
                    # Download button for markdown
                    markdown_content = (
                        f"# {content.get('title')}\n\n"
                        f"## Executive Summary\n"
                        f"{content.get('summary')}\n\n"
                        f"## Key Findings\n"
                        + "\n".join(
                            f"- {finding}" for finding in content.get("key_findings","N/A")
                        )
                        + "\n\n"
                        f"## Analysis\n"
                        f"{content.get('content')}\n\n"
                        f"## Key Entity Relationships\n"
                        + "\n".join(
                            f"### {entity.get('entity')}\n"
                            + "\n".join(f"- {r}" for r in entity.get("relationships"))
                            for entity in content.get("entity_relationships","N/A")
                        )
                        + "\n\n"
                        f"## References\n"
                        + "\n".join(
                            f"{ref['id']}. [{ref['title']}]({ref['link']}) - {ref['source']}, {ref['date']}"
                            for ref in content["references"]
                        )
                    )

                    st.download_button(
                        label="📥 Download as Markdown",
                        data=markdown_content,
                        file_name="consolidated_news.md",
                        mime="text/markdown",
                    )

                st.markdown("## Executive Summary")
                st.markdown(content["summary"])

                st.markdown("## Key Findings")
                for finding in content["key_findings"]:
                    st.markdown(f"- {finding}")

                st.markdown("## Analysis")
                st.markdown(content["content"])

                st.markdown("## Key Entity Relationships")
                for entity in content["entity_relationships"]:
                    st.markdown(f"### {entity['entity']}")
                    for relationship in entity["relationships"]:
                        st.markdown(f"- {relationship}")

                st.markdown("## References")
                for ref in content["references"]:
                    st.markdown(
                        f"{ref['id']}. [{ref['title']}]({ref['link']}) - "
                        f"{ref['source']}, {ref['date']}"
                    )
    else:
        st.sidebar.warning("Chat history is empty. Ask some questions first!")


# Clear chat button in sidebar
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.suggested_questions = []
    st.session_state.selected_article = None
    st.rerun()
