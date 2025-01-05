import snowflake.connector
import logging
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import streamlit as st
import json
import uuid
import re
from util import json_cleanup, get_relevant_charts

logger = logging.getLogger(__name__)


class SnowflakeOperations:
    def __init__(self):
        """Initialize Snowflake connection parameters from environment variables"""
        self.conn_params = {
            "user": st.secrets["snowflake"]["user"],
            "password": st.secrets["snowflake"]["password"],
            "account": st.secrets["snowflake"]["account"],
            "warehouse": st.secrets["snowflake"]["warehouse"],
            "database": st.secrets["snowflake"]["database"],
            "schema": st.secrets["snowflake"]["schema"],
        }

        # Create table if it doesn't exist
        self._create_news_table()

    def _get_connection(self):
        """Create and return a Snowflake connection"""
        try:
            return snowflake.connector.connect(**self.conn_params)
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            raise

    def _create_news_table(self):
        """Create the stock_news table if it doesn't exist"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS stock_news (
            article_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            title VARCHAR,
            publication_date TIMESTAMP,
            source VARCHAR,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
            link TEXT,
            PRIMARY KEY (article_id)
        )
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
                cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
                cursor.execute(create_table_query)
                logger.info("Successfully created/verified stock_news table")
        except Exception as e:
            logger.error(f"Failed to create stock_news table: {str(e)}")
            raise

    def insert_news_articles(self, articles: List[Dict]):
        """
        Insert news articles into Snowflake using MERGE to handle duplicates

        Args:
            articles: List of dictionaries containing article information
        """
        merge_query = """
        MERGE INTO stock_news t
        USING (
            SELECT
                %s as article_id,
                %s as symbol,
                %s as title,
                %s as publication_date,
                %s as source,
                %s as content,
                %s as link,
                CURRENT_TIMESTAMP() as created_at
        ) s
        ON t.article_id = s.article_id
        WHEN NOT MATCHED THEN
            INSERT (article_id, symbol, title, publication_date, source, content, link, created_at)
            VALUES (s.article_id, s.symbol, s.title, s.publication_date, s.source, s.content, s.link, s.created_at)
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
                cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")

                articles_processed = 0
                articles_skipped = 0

                logger.info(f"Starting to process {len(articles)} articles using MERGE")

                for article in articles:
                    try:
                        article_id = article["article_id"]
                        # Execute MERGE for each article
                        cursor.execute(
                            merge_query,
                            (
                                article_id,
                                article["symbol"],
                                article["title"],
                                article["publication_date"],
                                article["source"],
                                article["content"],
                                article["link"],
                            ),
                        )

                        # Check if MERGE resulted in an insert
                        rows_inserted = cursor.rowcount
                        if rows_inserted > 0:
                            articles_processed += 1
                            logger.debug(
                                f"Inserted new article: {article['title'][:100]} "
                                f"(ID: {article_id})"
                            )
                        else:
                            articles_skipped += 1
                            logger.debug(
                                f"Skipped duplicate article: {article['title'][:100]} "
                                f"(ID: {article_id})"
                            )

                        if (articles_processed + articles_skipped) % 10 == 0:
                            logger.info(
                                f"Progress: Processed {articles_processed} articles, "
                                f"Skipped {articles_skipped} duplicates"
                            )

                    except Exception as e:
                        logger.error(
                            f"Error processing article '{article.get('title', 'Unknown')}': "
                            f"{str(e)}"
                        )
                        continue

                conn.commit()
                logger.info(
                    f"Article MERGE completed:\n"
                    f"- Successfully inserted: {articles_processed}\n"
                    f"- Skipped duplicates: {articles_skipped}\n"
                    f"- Total processed: {articles_processed + articles_skipped}"
                )

        except Exception as e:
            logger.error(f"Failed to merge articles batch: {str(e)}")
            raise

    def get_sample_news(self, limit: int = 5) -> List[Dict]:
        """
        Retrieve a sample of recent news articles

        Args:
            limit: Number of articles to retrieve

        Returns:
            List of dictionaries containing article information
        """
        query = """
        SELECT article_id, symbol, title, publication_date, source
        FROM stock_news
        ORDER BY publication_date DESC
        LIMIT %s
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
                cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
                cursor.execute(query, (limit,))

                columns = [
                    "article_id",
                    "symbol",
                    "title",
                    "publication_date",
                    "source",
                ]
                results = []

                for row in cursor:
                    results.append(dict(zip(columns, row)))

                return results
        except Exception as e:
            logger.error(f"Failed to retrieve sample news: {str(e)}")
            raise

    def get_news_by_symbol(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Retrieve news articles for a specific symbol within the last N days

        Args:
            symbol: Stock symbol
            days: Number of days to look back

        Returns:
            List of dictionaries containing article information
        """
        query = """
        SELECT article_id, symbol, title, publication_date, source, content,link
        FROM stock_news
        WHERE symbol = %s
        AND publication_date >= DATEADD('day', -%s, CURRENT_TIMESTAMP())
        ORDER BY publication_date DESC
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
                cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
                cursor.execute(query, (symbol, days))

                columns = [
                    "article_id",
                    "symbol",
                    "title",
                    "publication_date",
                    "source",
                    "content",
                    "link",
                ]
                results = []

                for row in cursor:
                    results.append(dict(zip(columns, row)))
                # print(results)
                return results
        except Exception as e:
            logger.error(f"Failed to retrieve news for {symbol}: {str(e)}")
            raise

    def update_last_run(self, process_name: str, status: str, records_processed: int):
        """Update the last run timestamp for a process"""
        upsert_query = """
        MERGE INTO sync_control t
        USING (SELECT %s as process_name, CURRENT_TIMESTAMP() as last_run_timestamp, 
            %s as status, %s as records_processed) s
        ON t.process_name = s.process_name
        WHEN MATCHED THEN 
            UPDATE SET 
                last_run_timestamp = s.last_run_timestamp,
                status = s.status,
                records_processed = s.records_processed,
                updated_at = CURRENT_TIMESTAMP()
        WHEN NOT MATCHED THEN
            INSERT (process_name, last_run_timestamp, status, records_processed)
            VALUES (s.process_name, s.last_run_timestamp, s.status, s.records_processed)
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
                cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
                cursor.execute(upsert_query, (process_name, status, records_processed))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update last run timestamp: {str(e)}")
            raise

    def get_last_run(self, process_name: str) -> datetime:
        """
        Get the last run timestamp for a process.
        If no previous run exists, returns date from 1 month ago.
        """
        query = """
        SELECT max(last_run_timestamp)
        FROM sync_control
        WHERE process_name = %s
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
                cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
                cursor.execute(query, (process_name,))
                result = cursor.fetchone()

                if result and result[0]:
                    logger.info(
                        f"Previous run found for {process_name}. Last date: {result[0]}"
                    )
                    return result[0]
                else:
                    # If no previous run, return date from 1 month ago
                    cursor.execute("SELECT DATEADD('month', -1, CURRENT_TIMESTAMP())")
                    default_date = cursor.fetchone()[0]
                    logger.info(
                        f"No previous run found for {process_name}. Using default date: {default_date}"
                    )
                    return default_date

        except Exception as e:
            logger.error(f"Failed to get last run timestamp: {str(e)}")
            raise

    def _generate_standardized_id(self, entity_type: str, name: str) -> str:
        """Generate a standardized ID for an entity based on its type and name"""
        base_name = name.lower().replace(" ", "-").replace("/", "-")
        current_date = datetime.now().strftime("%Y%m%d")

        if entity_type == "COMPANY":
            # Extract ticker if it's in parentheses
            ticker_match = re.search(r"\(([^)]+)\)", name)
            if ticker_match:
                return f"company-{ticker_match.group(1).lower()}"
            return f"company-{base_name}"

        elif entity_type == "PERSON":
            return f"person-{base_name}"

        elif entity_type == "PRODUCT":
            return f"product-{base_name}"

        elif entity_type == "EVENT":
            return f"event-{base_name}-{current_date}"

        elif entity_type == "TOPIC":
            return f"topic-{base_name}"

        return f"{entity_type.lower()}-{base_name}-{str(uuid.uuid4())[:8]}"

    def extract_entities_and_relationships(self, article_id: str) -> str:
        """Extract entities and relationships from an article using Mistral"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
                cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
                # Get article content
                cursor.execute(
                    """
                    SELECT symbol, title, content
                    FROM stock_news
                    WHERE article_id = %s
                """,
                    (article_id,),
                )

                article = cursor.fetchone()
                # print(f"Article name {article}")
                if not article:
                    return "Article not found"

                symbol, title, content = article

                # Prepare prompt for Mistral
                prompt = f"""
                Given the following article about {symbol}, analyze and extract key entities and their relationships. Focus on significant, factual information only.

                    Article Title: {title}
                    Article Content: {content}

                    Instructions:
                    1. Entity Extraction Rules:
                    - For COMPANY: Include stock ticker if mentioned
                    - For PERSON: Include their role/position
                    - For PRODUCT: Include version/category
                    - For EVENT: Include approximate date/timeframe
                    - For TOPIC: Include domain/category
                    
                    2. Relationship Requirements:
                    - Only create relationships with high confidence (>0.7)
                    - Each relationship must be supported by explicit mentions in the text
                    - Avoid speculative or implied relationships

                    3. Valid Entity Types:
                    COMPANY, PERSON, PRODUCT, EVENT, TOPIC

                    4. Valid Relationship Types:
                    MENTIONS, PARTNERS_WITH, COMPETES_WITH, LAUNCHES, ACQUIRES, 
                    INVESTS_IN, LEADS, WORKS_FOR, PRODUCES, AFFECTS

                    Return in the following JSON structure:
                    {{
                        "entities": [
                            {{
                                "id": "unique_id",
                                "type": "COMPANY|PERSON|PRODUCT|EVENT|TOPIC",
                                "name": "entity_name",
                                "description": "detailed_contextual_description"
                            }}
                        ],
                        "relationships": [
                            {{
                                "source": "source_entity_id",
                                "target": "target_entity_id",
                                "type": "relationship_type",
                                "confidence": confidence_score,  # Must be between 0.7 and 1.0
                                "evidence": "relevant_text_snippet"  # Brief text supporting this relationship
                            }}
                        ]
                    }}

                    Note: 
                    - Ensure each entity has a meaningful description that provides context from the article
                    - Only include relationships that are explicitly mentioned in the text
                    - Entity IDs should be unique within this article's scope
                """

                # Call Mistral for extraction
                cursor.execute(
                    """
                SELECT VALIDATE_EXTRACTED_DATA(
                    SNOWFLAKE.CORTEX.COMPLETE(
                        'mistral-large2',
                        %s
                    )
                )
                """,
                    (prompt,),
                )
                result = json.loads(cursor.fetchone()[0])

                logger.info(f"extracted_data: {result}")
                logger.info(type(result))

                # Check if validation was successful
                if not result.get("validated", False):
                    raise ValueError(
                        f"Validation failed: {result.get('error', 'Unknown error')}"
                    )
                validated_data = result["data"]
                # Create a mapping of model-generated IDs to standardized IDs
                entity_id_map = {}
                logger.info("Process entities")
                # Process entities
                for entity in validated_data["entities"]:
                    logger.info(f"Process entitie: {entity}")
                    standardized_id = self._generate_standardized_id(
                        entity["type"], entity["name"]
                    )
                    entity_id_map[entity["id"]] = standardized_id
                    entity_text = f'{entity["name"]} {entity["description"]}'

                    # Generate embedding
                    cursor.execute(
                        """
                        SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
                            'snowflake-arctic-embed-m-v1.5',
                            %s
                        )
                    """,
                        (entity_text,),
                    )

                    embedding_result = cursor.fetchone()[0]
                    logger.info(f'Embedding completed {entity["name"]}')
                    # Insert node
                    # cursor.execute(
                    #     f"""
                    #         INSERT INTO nodes
                    #         SELECT
                    #             '{standardized_id}',
                    #             '{entity["type"].replace("'","")}',
                    #             '{entity["name"].replace("'","")}',
                    #             '{entity["description"].replace("'","")}',
                    #             Current_Timestamp,
                    #             {embedding_result}::VECTOR(FLOAT, 768),
                    #             '{article_id}'
                    #             ON CONFLICT (node_id) DO UPDATE SET
                    #             description = EXCLUDED.description,
                    #             updated_at = Current_Timestamp
                    #     """
                    # )
                    cursor.execute(
                        f"""
                        MERGE INTO nodes t
                        USING (
                            SELECT 
                                '{standardized_id}' as node_id,
                                '{entity["type"].replace("'","")}' as node_type,
                                '{entity["name"].replace("'","")}' as name,
                                '{entity["description"].replace("'","")}' as description,
                                Current_Timestamp as created_at,
                                {embedding_result}::VECTOR(FLOAT, 768) as embedding,
                                '{article_id}' as source_articles
                        ) s
                        ON t.node_id = s.node_id
                        WHEN MATCHED THEN
                            UPDATE SET
                                description = s.description,
                                created_at = s.created_at
                        WHEN NOT MATCHED THEN
                            INSERT (node_id, node_type, name, description, created_at, embedding, source_articles)
                            VALUES (s.node_id, s.node_type, s.name, s.description, s.created_at, s.embedding, s.source_articles)
                    """
                    )
                logger.info("Process relationships")
                # Process relationships
                for rel in validated_data["relationships"]:
                    logger.info(f"Process relationship: {rel}")
                    rel_id = str(uuid.uuid4())
                    source_id = entity_id_map[rel["source"]]
                    target_id = entity_id_map[rel["target"]]
                    cursor.execute(
                        """
                        INSERT INTO relationships (
                            relationship_id, source_node_id, target_node_id, 
                            relationship_type, confidence, evidence,source_articles
                        )
                        VALUES (%s, %s, %s, %s, %s,%s, %s)
                    """,
                        (
                            rel_id,
                            source_id,
                            target_id,
                            rel["type"],
                            rel["confidence"],
                            rel["evidence"],
                            article_id,
                        ),
                    )

                return "Successfully processed article"

        except Exception as e:
            logger.error(f"Error processing article {article_id}: {str(e)}")
            raise

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on nodes"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
                cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
                # Get query embedding
                cursor.execute(
                    """
                    WITH query_embedding AS (
                        SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
                            'snowflake-arctic-embed-m-v1.5',
                            %s
                        ) as qemb
                    )
                    SELECT 
                        n.node_id,
                        n.node_type,
                        n.name,
                        n.description,
                        VECTOR_COSINE_SIMILARITY(n.embedding, qemb) as similarity
                    FROM nodes n, query_embedding
                    ORDER BY similarity DESC
                    LIMIT %s
                """,
                    (query, top_k),
                )

                columns = ["node_id", "node_type", "name", "description", "similarity"]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise

    def get_node_relationships(self, node_id: str) -> Dict[str, List[Dict]]:
        """Get all relationships for a node"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
                cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
                # Get outgoing relationships
                cursor.execute(
                    """
                    SELECT 
                        r.*,
                        n.name as target_name,
                        n.node_type as target_type,
                        n.source_articles as article_id,
                        sn.link as article_link,
                        sn.publication_date,
                        sn.source as article_source
                    FROM relationships r
                    JOIN nodes n ON r.target_node_id = n.node_id
                    LEFT JOIN stock_news sn ON n.source_articles = sn.article_id
                    WHERE r.source_node_id = %s
                    ORDER BY sn.publication_date DESC;
                """,
                    (node_id,),
                )

                outgoing = [
                    dict(zip([d[0] for d in cursor.description], row))
                    for row in cursor.fetchall()
                ]

                # Get incoming relationships
                cursor.execute(
                    """
                    SELECT r.*,
                        n.name as target_name,
                        n.node_type as target_type,
                        n.source_articles as article_id,
                        sn.link as article_link,
                        sn.publication_date,
                        sn.source as article_source
                    FROM relationships r
                    JOIN nodes n ON r.source_node_id = n.node_id
                    LEFT JOIN stock_news sn ON n.source_articles = sn.article_id
                    WHERE r.target_node_id = %s
                    ORDER BY sn.publication_date DESC;
                """,
                    (node_id,),
                )

                incoming = [
                    dict(zip([d[0] for d in cursor.description], row))
                    for row in cursor.fetchall()
                ]

                return {"outgoing": outgoing, "incoming": incoming}

        except Exception as e:
            logger.error(f"Error getting relationships for node {node_id}: {str(e)}")
            raise

    def query_graph(self, question: str) -> str:
        """Query the knowledge graph using Mistral LLM"""
        try:
            # First get relevant nodes
            relevant_nodes = self.semantic_search(question, top_k=3)
            logger.info("relevant_nodes completed")

            # Get relationships for each relevant node
            context = []
            for node in relevant_nodes:
                relationships = self.get_node_relationships(node["node_id"])
                # relationships = relationships.pop("CREATED_AT")
                context.append({"node": node, "relationships": relationships})
            # print(f"context {context}")
            return context

            # # Prepare prompt for Mistral
            # prompt = f"""
            # Based on the following knowledge graph information, answer this question:
            # Question: {question}

            # Context:
            # {context}

            # Answer the question based only on the information provided in the context.
            # If the information is not sufficient, say so.
            # """

            # logger.info(f"prompt {prompt}")
            # with self._get_connection() as conn:
            #     cursor = conn.cursor()
            #     cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
            #     cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
            #     # Query Mistral
            #     cursor.execute(
            #         """
            #         SELECT SNOWFLAKE.CORTEX.COMPLETE(
            #             'mistral-large2',
            #             %s
            #         )
            #     """,
            #         (prompt,),
            #     )

            #     return cursor.fetchone()[0]

        except Exception as e:
            logger.error(f"Error querying graph: {str(e)}")
            raise

    def build_incremental_graph(self) -> str:
        """Build the knowledge graph incrementally from unprocessed articles"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
                cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")

                # Get last processed timestamp
                last_run = self.get_last_run("graph_build")

                # Get unprocessed articles
                logger.info(
                    """
                        SELECT DISTINCT sn.article_id
                        FROM stock_news sn
                        LEFT JOIN nodes n ON n.source_articles = sn.article_id
                        WHERE n.node_id IS NULL
                        OR sn.created_at > %s
                    """,
                    (last_run,),
                )
                if last_run:
                    cursor.execute(
                        """
                        SELECT DISTINCT sn.article_id
                        FROM stock_news sn
                        LEFT JOIN nodes n ON n.source_articles = sn.article_id
                        WHERE n.node_id IS NULL
                        OR sn.created_at > %s
                    """,
                        (last_run,),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT DISTINCT sn.article_id
                        FROM stock_news sn
                        LEFT JOIN nodes n ON n.source_articles = sn.article_id
                        WHERE n.node_id IS NULL
                    """
                    )

                article_ids = [row[0] for row in cursor.fetchall()]
                logger.info(f"List of article_ids {article_ids}")
                processed = 0
                failed = 0

                for article_id in article_ids:
                    try:
                        # Use existing function to process each article
                        self.extract_entities_and_relationships(article_id)
                        processed += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to process article {article_id}: {str(e)}"
                        )
                        failed += 1
                        continue

                # Update last run timestamp
                self.update_last_run("graph_build", "completed", processed)

                return f"Processed {processed} articles, {failed} failed"

        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            raise

    def get_last_cleanup(self) -> datetime:
        """Get last cache cleanup timestamp"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
            cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
            cursor.execute(
                """
                SELECT MAX(last_run_timestamp)
                FROM sync_control
                WHERE process_name = 'cache_cleanup'
            """
            )
            result = cursor.fetchone()[0]
            return result if result else datetime.min

    def update_cleanup_timestamp(self):
        """Update cache cleanup timestamp"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
            cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
            cursor.execute(
                """
                INSERT INTO sync_control (process_name, last_run_timestamp, status)
                VALUES ('cache_cleanup', CURRENT_TIMESTAMP(), 'completed')
                ON CONFLICT (process_name) DO UPDATE 
                SET last_run_timestamp = CURRENT_TIMESTAMP()
            """
            )

    def cache_query_response(
        self, query: str, response: dict, is_pro_search: bool = False
    ):
        import hashlib

        query_hash = hashlib.md5(query.lower().encode()).hexdigest()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
            cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
            cursor.execute(
                """
                    MERGE INTO query_cache target
                    USING (SELECT column1 as query_hash, column2 as query_text, 
                                PARSE_JSON(column3) as response 
                        FROM VALUES(%s, %s, %s)) source
                    ON target.query_hash = source.query_hash
                    WHEN MATCHED THEN 
                        UPDATE SET 
                            access_count = target.access_count + 1,
                            last_accessed = CURRENT_TIMESTAMP()
                    WHEN NOT MATCHED THEN
                        INSERT (query_hash, query_text, response)
                        VALUES (source.query_hash, source.query_text, source.response)
                """,
                (query_hash, query, json.dumps(response)),
            )

    def generate_refined_queries(self, query: str) -> List[str]:
        """Generate refined versions of the user query with step-by-step breakdown"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
                cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")

                prompt = f"""
                        Given this user question about stock market news:
                        "{query}"
                        
                        1. First, break down this question into key components and aspects that need to be analyzed.
                        2. Then, generate only 2 refined versions of this question that:
                            a. Make it more specific and detailed
                            b. Consider related market factors or entities
                        
                        Return the response in this JSON format:
                        {{
                            "breakdown": [
                                "step 1 of analysis",
                                "step 2 of analysis",
                                ...
                            ],
                            "refined_questions": [
                                "refined question 1",
                                "refined question 2",
                            ]
                        }}
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
                parsed_result = json.loads(result)

                return parsed_result

        except Exception as e:
            st.error(f"Error generating refined queries: {str(e)}")
            return {"breakdown": [], "refined_questions": [query]}

    def get_trending_questions(self, limit: int = 5) -> list:
        """
        Get the most frequently asked questions from the query cache

        Args:
            limit: Number of trending questions to return

        Returns:
            List of dict containing query and frequency
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
                cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")

                # Query to get trending questions from last 7 days
                cursor.execute(
                    """
                    SELECT 
                        query_text,
                        query_hash,
                        MAX(access_count) as frequency,
                    FROM QUERY_CACHE 
                    WHERE last_accessed >= DATEADD(day, -7, CURRENT_TIMESTAMP())
                    GROUP BY query_text,query_hash
                    ORDER BY frequency DESC
                    LIMIT %s
                """,
                    (limit,),
                )

                results = cursor.fetchall()

                trending = []
                for query, query_hash, freq in results:
                    trending.append(
                        {"query": query, "frequency": freq, "query_hash": query_hash}
                    )

                return trending

        except Exception as e:
            logger.error(f"Error fetching trending questions: {str(e)}")
            return []

    def generate_consolidated_contents(self, messages: List[Dict]) -> Dict:
        """Generate consolidated content from chat history with references"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
                cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")

                # Extract messages and charts
                relevant_messages = [
                    msg
                    for msg in messages
                    if msg.get("role") == "assistant" and "references" in msg
                ]
                relevant_charts = get_relevant_charts(messages)

                # Sanitize and prepare context
                def sanitize_text(text):
                    if isinstance(text, str):
                        # Remove control characters and ensure valid JSON string
                        return ' '.join(text.split()).replace('\n', ' ').replace('\r', ' ')
                    return text

                def sanitize_dict(d):
                    return {k: sanitize_text(v) if isinstance(v, str) else v for k, v in d.items()}

                # Prepare sanitized context
                context = {
                    "messages": [
                        {
                            "content": sanitize_text(msg["content"]),
                            "references": [sanitize_dict(ref) for ref in msg["references"]],
                            "key_entities": [sanitize_dict(entity) for entity in msg.get("key_entities", [])]
                        }
                        for msg in relevant_messages
                    ],
                    "charts": [
                        {
                            "title": sanitize_text(chart["title"]),
                            "description": sanitize_text(chart["description"]),
                            "type": chart["type"]
                        }
                        for chart in relevant_charts
                    ]
                }

                prompt = f"""
                Based on these chat messages, references, and visualizations, generate a consolidated article that:
                1. Synthesizes the key insights and findings
                2. Maintains all source citations in markdown format
                3. Groups related topics coherently
                4. Integrates relevant charts and visualizations appropriately
                5. Includes a summary section at the start
                6. Lists key entities and their relationships
                
                Chat Context: {json.dumps(context)}
                
                Return the response in this JSON format:
                {{
                    "title": "Generated title based on content",
                    "summary": "Executive summary paragraph",
                    "content": "Full markdown content with [n] citations",
                    "key_findings": ["List of key findings"],
                    "visualizations": [
                        {{
                            "chart_index": n,  # Index of chart in original list
                            "placement": "Description of where to place in content",
                            "context": "Additional context for this visualization"
                        }}
                    ],
                    "entity_relationships": [
                        {{
                            "entity": "entity_name",
                            "relationships": ["relationship descriptions"]
                        }}
                    ],
                    "references": [
                        {{
                            "id": "citation_number",
                            "title": "source_title",
                            "source": "source_name",
                            "date": "publication_date",
                            "link": "article_link"
                        }}
                    ]
                }}
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

                result = json.loads(json_cleanup(cursor.fetchone()[0]))

                # Add charts to result
                result["charts"] = relevant_charts
                return result

        except Exception as e:
            st.error(f"Error generating consolidated content: {str(e)}")
            return None

    def show_article_contents(self, link: str, ref_idx: str):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
                cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
                cursor.execute(
                    """
                    SELECT content 
                    FROM stock_news 
                    WHERE link = %s
                    """,
                    (link,),
                )
                result = cursor.fetchone()

                if result and result[0]:
                    return result[0]
                else:
                    st.warning("Content not found in database")
                    st.markdown(f"[Read article on source website]({link})")
        except Exception as e:
            st.error(f"Error fetching article content: {str(e)}")

    def get_cached_response(self, query: str) -> dict:
        """Get cached response if less than 24 hours old"""
        import hashlib

        query_hash = hashlib.md5(query.lower().encode()).hexdigest()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
            cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
            cursor.execute(
                """
                SELECT response 
                FROM query_cache 
                WHERE query_hash = %s 
                AND created_at > DATEADD(hour, -24, CURRENT_TIMESTAMP())
            """,
                (query_hash,),
            )

            result = cursor.fetchone()
            if result:
                # Update access count
                cursor.execute(
                    """
                    UPDATE query_cache 
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP()
                    WHERE query_hash = %s
                """,
                    (query_hash,),
                )
                return result[0]
            return None

    def cleanup_cache(self):
        """Remove cached entries older than 24 hours"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"USE DATABASE \"{self.conn_params['database']}\"")
            cursor.execute(f"USE SCHEMA \"{self.conn_params['schema']}\"")
            cursor.execute(
                """
                DELETE FROM query_cache 
                WHERE created_at < DATEADD(hour, -24, CURRENT_TIMESTAMP())
            """
            )
