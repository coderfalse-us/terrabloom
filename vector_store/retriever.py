import numpy as np
import pandas as pd
import re
from typing import List, Dict, Set, Tuple
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from vector_store.ivf_faiss_store import IVFFAISSStore
from config.config import config

class IVFFAISSRetriever:
    """Enhanced LangChain-compatible retriever for IVF FAISS store with better join support"""
    
    def __init__(self, store: IVFFAISSStore = None):
        self.store = store
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
        self.table_relationships = {}  # Cache for table relationships
        
        # Load existing store if available
        if store is None:
            self.store = self._load_or_create_store()
    
    def _load_or_create_store(self) -> IVFFAISSStore:
        """Load existing store or create new one"""
        print(f"Attempting to load schema store from: {config.SCHEMA_STORE_PATH}")
        try:
            # Try to load existing store
            store = IVFFAISSStore.load(config.SCHEMA_STORE_PATH)
            print(f"Successfully loaded schema store with {store.id_counter} documents")
            print(f"Index size: {store.index.ntotal if store.index else 'No index'}")
            print(f"Document store size: {len(store.document_store)}")
            print(f"Metadata store size: {len(store.metadata_store)}")
            
            self._build_relationship_cache()
            return store
        except Exception as e:
            print(f"ERROR loading schema store: {str(e)}")
            import traceback
            print(traceback.format_exc())
            print("Creating new schema store...")
            return self._create_schema_store()
    
    def _extract_relationships(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Extract potential relationships between tables"""
        relationships = {}
        table_columns = {}
        
        # First pass: collect all columns per table
        for _, row in df.iterrows():
            table_name = row['table_name']
            columns = self._extract_columns_from_ddl(row['DDL'])
            table_columns[table_name] = columns
        
        # Second pass: find potential relationships
        for table_name, columns in table_columns.items():
            relationships[table_name] = {
                'foreign_keys': [],
                'referenced_by': [],
                'potential_joins': []
            }
            
            for col_name, col_type in columns:
                # Look for foreign key patterns
                if col_name.endswith('id') and col_name != 'id':
                    # Try to find referenced table
                    potential_table = col_name[:-2]  # Remove 'id' suffix
                    for other_table in table_columns:
                        if (potential_table.lower() in other_table.lower() or 
                            other_table.lower() in potential_table.lower()):
                            relationships[table_name]['foreign_keys'].append({
                                'column': col_name,
                                'references_table': other_table,
                                'references_column': 'id'
                            })
                
                # Look for columns that might be referenced by others
                if col_name == 'id':
                    for other_table, other_columns in table_columns.items():
                        if other_table != table_name:
                            for other_col, _ in other_columns:
                                # Check if other table has a column that might reference this table
                                table_base = table_name.split('.')[-1] if '.' in table_name else table_name
                                if other_col == f"{table_base}id" or other_col == f"{table_base}_id":
                                    relationships[table_name]['referenced_by'].append({
                                        'table': other_table,
                                        'column': other_col
                                    })
        
        return relationships
    
    def _build_relationship_cache(self):
        """Build cache of table relationships from existing store"""
        print("Building relationship cache from metadata...")
        try:
            # Extract relationships from metadata
            relationships = {}
            
            # Count tables and relationships for debugging
            table_count = 0
            relationship_count = 0
            
            # Process metadata to rebuild relationships
            for doc_id, metadata in self.store.metadata_store.items():
                if metadata.get('type') == 'table_schema':
                    table_count += 1
                    table_name = metadata.get('table')
                    if table_name and 'relationships' in metadata:
                        relationships[table_name] = metadata['relationships']
                        relationship_count += len(metadata['relationships'].get('foreign_keys', []))
                        relationship_count += len(metadata['relationships'].get('referenced_by', []))
            
            self.table_relationships = relationships
            print(f"Relationship cache built with {table_count} tables and {relationship_count} relationships")
            
            # Print a sample of relationships for debugging
            if relationships:
                sample_table = next(iter(relationships))
                print(f"Sample relationship for table {sample_table}: {relationships[sample_table]}")
            else:
                print("WARNING: No relationships found in metadata!")
                
        except Exception as e:
            print(f"ERROR building relationship cache: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.table_relationships = {}
    
    def _extract_columns_from_ddl(self, ddl: str) -> List[Tuple[str, str]]:
        """Extract column definitions from DDL"""
        ddl_lines = ddl.split('\n')
        columns = []
        
        for line in ddl_lines:
            line = line.strip()
            # Skip non-column lines
            if (not line or 
                line.startswith('CREATE TABLE') or 
                line.startswith('(') or 
                line.startswith(')') or 
                line.startswith('CONSTRAINT') or
                line.startswith('PRIMARY KEY') or
                line.startswith('FOREIGN KEY') or
                line.startswith('UNIQUE') or
                line.startswith('CHECK') or
                line.endswith(';')):
                continue
            
            # Remove trailing comma and whitespace
            line = line.rstrip(',').strip()
            
            # Match column definition pattern
            column_match = re.match(r'^(\w+)\s+((?:varchar|int\d*|float\d*|numeric|timestamp|bpchar|text|boolean|date|time)(?:\(\d+(?:,\d+)?\))?)', line, re.IGNORECASE)
            
            if column_match:
                col_name = column_match.group(1)
                col_type = column_match.group(2)
                columns.append((col_name, col_type))
        
        return columns
    
    def _create_schema_store(self) -> IVFFAISSStore:
        """Create an IVF FAISS store from schema data with relationship awareness"""
        # Load schema data
        df = pd.read_csv(config.TABLE_SCHEMA_PATH)
        
        # Extract relationships
        relationships = self._extract_relationships(df)
        self.table_relationships = relationships
        
        documents = []
        metadata = []
        
        for i, row in df.iterrows():
            columns = self._extract_columns_from_ddl(row['DDL'])
            
            # Extract schema and table name
            schema_name, table_name = row['table_name'].split('.') if '.' in row['table_name'] else ('public', row['table_name'])
            
            # Get relationship information for this table
            table_rels = relationships.get(row['table_name'], {})
            
            # Build join context
            join_context_parts = []
            
            # Add foreign key information
            if table_rels.get('foreign_keys'):
                fk_info = []
                for fk in table_rels['foreign_keys']:
                    fk_info.append(f"{fk['column']} -> {fk['references_table']}.{fk['references_column']}")
                join_context_parts.append(f"FOREIGN_KEYS: {', '.join(fk_info)}")
            
            # Add reverse relationship information
            if table_rels.get('referenced_by'):
                ref_info = []
                for ref in table_rels['referenced_by']:
                    ref_info.append(f"{ref['table']}.{ref['column']} -> id")
                join_context_parts.append(f"REFERENCED_BY: {', '.join(ref_info)}")
            
            join_context = '\n' + '\n'.join(join_context_parts) if join_context_parts else ''
            
            # Create comprehensive table document with relationship info
            table_content = f"""TABLE: {row['table_name']}
SCHEMA: {schema_name}
TABLE_NAME: {table_name}
COLUMNS: {', '.join([f"{col[0]} ({col[1]})" for col in columns])}{join_context}
FULL_DDL: {row['DDL']}"""
            
            table_doc = {
                'content': table_content,
                'type': 'table_schema'
            }
            
            table_meta = {
                'table': row['table_name'],
                'schema': schema_name,
                'table_name': table_name,
                'type': 'table_schema',
                'column_count': len(columns),
                'relationships': table_rels
            }
            
            documents.append(table_doc)
            metadata.append(table_meta)
            
            # Create enhanced documents for each column with relationship context
            for col_name, col_type in columns:
                # Add relationship context for this column
                col_relationship_context = ""
                
                # Check if this column is a foreign key
                for fk in table_rels.get('foreign_keys', []):
                    if fk['column'] == col_name:
                        col_relationship_context += f"\nFOREIGN_KEY: References {fk['references_table']}.{fk['references_column']}"
                
                # Check if this column is referenced by others
                if col_name == 'id':
                    for ref in table_rels.get('referenced_by', []):
                        col_relationship_context += f"\nREFERENCED_BY: {ref['table']}.{ref['column']}"
                
                col_content = f"""TABLE: {row['table_name']}
COLUMN: {col_name}
DATA_TYPE: {col_type}
SCHEMA: {schema_name}
TABLE_NAME: {table_name}
CONTEXT: Column {col_name} of type {col_type} in table {row['table_name']}{col_relationship_context}"""
                
                col_doc = {
                    'content': col_content,
                    'type': 'column_info'
                }
                
                col_meta = {
                    'table': row['table_name'],
                    'schema': schema_name,
                    'table_name': table_name,
                    'column': col_name,
                    'type': 'column_info',
                    'data_type': col_type,
                    'is_foreign_key': any(fk['column'] == col_name for fk in table_rels.get('foreign_keys', [])),
                    'is_referenced': col_name == 'id' and bool(table_rels.get('referenced_by'))
                }
                
                documents.append(col_doc)
                metadata.append(col_meta)
            
            # Create join-specific documents for better retrieval of relationship queries
            for fk in table_rels.get('foreign_keys', []):
                join_content = f"""JOIN_RELATIONSHIP: {row['table_name']} -> {fk['references_table']}
SOURCE_TABLE: {row['table_name']}
SOURCE_COLUMN: {fk['column']}
TARGET_TABLE: {fk['references_table']}
TARGET_COLUMN: {fk['references_column']}
JOIN_TYPE: Foreign Key Relationship
QUERY_PATTERNS: customer for {table_name}, {table_name} customer, join {row['table_name']} {fk['references_table']}"""
                
                join_doc = {
                    'content': join_content,
                    'type': 'join_relationship'
                }
                
                join_meta = {
                    'source_table': row['table_name'],
                    'target_table': fk['references_table'],
                    'source_column': fk['column'],
                    'target_column': fk['references_column'],
                    'type': 'join_relationship'
                }
                
                documents.append(join_doc)
                metadata.append(join_meta)
        
        # Get embeddings for all documents
        texts = [doc['content'] for doc in documents]
        embeddings = np.array(self.embeddings_model.embed_documents(texts))
        
        # Create IVF store
        store = IVFFAISSStore(embedding_dim=embeddings.shape[1], nlist=50)
        store.add_documents(embeddings, documents, metadata)
        
        # Save the store
        store.save(config.SCHEMA_STORE_PATH)
        
        return store
    
    def _detect_join_query(self, query: str) -> bool:
        """Detect if a query is likely about table joins"""
        print(f"Detecting if '{query}' is a join query...")
        
        join_keywords = [
            'join', 'relationship', 'related', 'connect', 'linked', 'association',
            'foreign key', 'references', 'between', 'link'
        ]
        
        query_lower = query.lower()
        
        # Check for join keywords
        for keyword in join_keywords:
            if keyword in query_lower:
                print(f"Join keyword '{keyword}' found in query")
                return True
        
        # Check for multiple table mentions
        table_mentions = []
        for table in self.table_relationships.keys():
            if table.lower() in query_lower:
                table_mentions.append(table)
                print(f"Table '{table}' mentioned in query")
                
        if len(table_mentions) >= 2:
            print(f"Multiple tables mentioned: {table_mentions}")
            return True
        
        print("Not detected as a join query")
        return False
    
    def invoke(self, query: str, k: int = 10) -> List[Document]:
        """Enhanced invoke method with better join query handling"""
        print(f"\n=== IVFFAISSRetriever invoke called with query: '{query}' and k={k} ===")
        
        try:
            # Get query embedding
            print("Embedding query...")
            query_embedding = np.array(self.embeddings_model.embed_query(query))
            print(f"Query embedding shape: {len(query_embedding)}")
            
            # Determine if this is likely a join query
            is_join_query = self._detect_join_query(query)
            print(f"Is join query: {is_join_query}")
            
            # Search with more results to allow filtering
            search_k = k * 3 if is_join_query else k * 2
            print(f"Using search_k={search_k}")
            
            print("Calling store.search...")
            results = self.store.search(query_embedding, k=search_k)
            print(f"Search returned {len(results)} results")
            
            if not results:
                print("WARNING: No results returned from search!")
                return []
                
            # Print first result for debugging
            if results:
                print(f"First result metadata: {results[0]['metadata']}")
                print(f"First result distance: {results[0]['distance']}")
                content_preview = results[0]['content'][:100] + '...' if len(results[0]['content']) > 100 else results[0]['content']
                print(f"First result content preview: {content_preview}")
            
            # Separate results by type
            table_results = []
            column_results = []
            join_results = []
            
            for result in results:
                result_type = result['metadata']['type']
                if result_type == 'table_schema':
                    table_results.append(result)
                elif result_type == 'column_info':
                    column_results.append(result)
                elif result_type == 'join_relationship':
                    join_results.append(result)
            
            documents = []
            
            if is_join_query:
                # For join queries, prioritize relationship information
                processed_relationships = set()
                
                # Add join relationship documents first
                for join_result in join_results[:3]:  # Limit to top 3 join relationships
                    rel_key = f"{join_result['metadata']['source_table']}-{join_result['metadata']['target_table']}"
                    if rel_key not in processed_relationships:
                        # Enhance join document with table information
                        source_table = join_result['metadata']['source_table']
                        target_table = join_result['metadata']['target_table']
                        
                        # Find corresponding table schemas
                        source_schema = next((r for r in table_results if r['metadata']['table'] == source_table), None)
                        target_schema = next((r for r in table_results if r['metadata']['table'] == target_table), None)
                        
                        enhanced_content = join_result['content']
                        if source_schema:
                            enhanced_content += f"\n\nSOURCE_TABLE_SCHEMA:\n{source_schema['content']}"
                        if target_schema:
                            enhanced_content += f"\n\nTARGET_TABLE_SCHEMA:\n{target_schema['content']}"
                        
                        documents.append(Document(
                            page_content=enhanced_content,
                            metadata=join_result['metadata']
                        ))
                        processed_relationships.add(rel_key)
                
                # Add relevant table schemas not already included
                processed_tables = {doc.metadata.get('source_table') for doc in documents} | \
                                 {doc.metadata.get('target_table') for doc in documents}
                
                for table_result in table_results:
                    if (len(documents) < k and 
                        table_result['metadata']['table'] not in processed_tables):
                        documents.append(Document(
                            page_content=table_result['content'],
                            metadata=table_result['metadata']
                        ))
            else:
                # For non-join queries, use original logic with improvements
                table_results_dict = {r['metadata']['table']: r for r in table_results}
                column_results_dict = {}
                
                for result in column_results:
                    table_name = result['metadata']['table']
                    if table_name not in column_results_dict:
                        column_results_dict[table_name] = []
                    column_results_dict[table_name].append(result)
                
                # Combine table and column information
                processed_tables = set()
                
                # Add table schemas with their relevant columns
                for table_name, table_result in table_results_dict.items():
                    if len(documents) >= k:
                        break
                        
                    content_parts = [table_result['content']]
                    
                    # Add relevant column information
                    if table_name in column_results_dict:
                        content_parts.append("\nRELEVANT_COLUMNS:")
                        for col_result in column_results_dict[table_name][:5]:
                            content_parts.append(f"- {col_result['metadata']['column']}: {col_result['metadata']['data_type']}")
                    
                    documents.append(Document(
                        page_content="\n".join(content_parts),
                        metadata=table_result['metadata']
                    ))
                    processed_tables.add(table_name)
                
                # Add remaining column results
                for table_name, col_results in column_results_dict.items():
                    if table_name not in processed_tables and len(documents) < k:
                        for col_result in col_results[:2]:
                            if len(documents) >= k:
                                break
                            documents.append(Document(
                                page_content=col_result['content'],
                                metadata=col_result['metadata']
                            ))
            print("Documents are", documents[:k])
            print("Number of documents:", len(documents))
            
            if len(documents) == 0:
                print("WARNING: No documents to return!")
                print("table_results:", len(table_results))
                print("column_results:", len(column_results))
                print("join_results:", len(join_results))
            else:
                # Print details of first document
                print("First document page_content type:", type(documents[0].page_content))
                print("First document page_content length:", len(documents[0].page_content))
                print("First document metadata:", documents[0].metadata)
            
            return documents[:k]
        except Exception as e:
            print(f"ERROR in invoke: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []
    
    def get_store_stats(self):
        """Get statistics about the vector store"""
        stats = self.store.get_stats()
        stats['relationships_count'] = len(self.table_relationships)
        return stats