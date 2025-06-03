"""Schema extraction utilities for the RAG application."""

import os
import logging
import pandas as pd
import psycopg2
from pathlib import Path
from config.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_table_schema(schema_name, table_name, connection):
    """
    Generate CREATE TABLE statement for a specific table.
    
    Args:
        schema_name (str): Database schema name
        table_name (str): Table name
        connection: Database connection object
        
    Returns:
        str: CREATE TABLE SQL statement
    """
    try:
        # Create a cursor to execute queries
        cursor = connection.cursor()

        # Query to get column details
        column_query = """
        SELECT
            column_name,
            udt_name,
            character_maximum_length,
            is_nullable
        FROM
            information_schema.columns
        WHERE
            table_schema = %s AND table_name = %s
        ORDER BY ordinal_position;
        """

        # Query to get primary key
        primary_key_query = """
        SELECT
            kcu.column_name
        FROM 
            information_schema.table_constraints tc
        JOIN 
            information_schema.key_column_usage kcu
        ON 
            tc.constraint_name = kcu.constraint_name
        WHERE 
            tc.table_schema = %s AND tc.table_name = %s AND tc.constraint_type = 'PRIMARY KEY'
        ORDER BY kcu.ordinal_position;
        """

        # Query to get foreign keys
        foreign_key_query = """
        SELECT
            kcu.column_name,
            ccu.table_schema AS foreign_table_schema,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM 
            information_schema.table_constraints AS tc 
        JOIN 
            information_schema.key_column_usage AS kcu
        ON 
            tc.constraint_name = kcu.constraint_name
        JOIN 
            information_schema.constraint_column_usage AS ccu
        ON 
            ccu.constraint_name = tc.constraint_name
        WHERE 
            tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = %s AND tc.table_name = %s;
        """

        # Execute queries
        cursor.execute(column_query, (schema_name, table_name))
        columns = cursor.fetchall()

        cursor.execute(primary_key_query, (schema_name, table_name))
        primary_keys = [row[0] for row in cursor.fetchall()]

        cursor.execute(foreign_key_query, (schema_name, table_name))
        foreign_keys = cursor.fetchall()

        # Build CREATE TABLE statement
        ddl = f"CREATE TABLE {schema_name}.{table_name} (\n"
        
        column_definitions = []
        for col in columns:
            col_name, data_type, max_length, is_nullable = col
            
            # Format data type
            if max_length and data_type in ['varchar', 'char']:
                type_def = f"{data_type}({max_length})"
            else:
                type_def = data_type
            
            # Add nullable constraint
            nullable = "" if is_nullable == 'YES' else " NOT NULL"
            
            column_definitions.append(f"    {col_name} {type_def}{nullable}")
        
        ddl += ",\n".join(column_definitions)
        
        # Add primary key constraint
        if primary_keys:
            ddl += f",\n    PRIMARY KEY ({', '.join(primary_keys)})"
        
        ddl += "\n);"
        
        return ddl
        
    except Exception as e:
        logger.error(f"Error extracting schema for {table_name}: {e}")
        return None
    finally:
        cursor.close()

def get_all_schemas(connection):
    """
    Get all schemas in the database (excluding system schemas).
    
    Args:
        connection: Database connection object
        
    Returns:
        list: List of schema names
    """
    try:
        cursor = connection.cursor()
        query = """
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
        AND schema_name NOT LIKE 'pg_temp_%'
        AND schema_name NOT LIKE 'pg_toast_temp_%'
        ORDER BY schema_name;
        """
        cursor.execute(query)
        #schemas = [row[0] for row in cursor.fetchall()]
        schemas=['automation', 'customersetup', 'document', 'inventorycontrol', 'item', 'orders','picking', 'receiving', 'returns', 'returnsmanagement', 'shipping', 'tracking', 'wave']
        cursor.close()
        return schemas
    except Exception as e:
        logger.error(f"Error getting schemas: {e}")
        return []

def get_all_tables(schema_name, connection):
    """
    Get all tables in a schema.
    
    Args:
        schema_name (str): Database schema name
        connection: Database connection object
        
    Returns:
        list: List of table names
    """
    try:
        cursor = connection.cursor()
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = %s 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
        cursor.execute(query, (schema_name,))
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return tables
    except Exception as e:
        logger.error(f"Error getting tables: {e}")
        return []

def extract_all_table_schemas():
    """
    Extract schemas for all tables in all schemas of the database.
    
    Returns:
        dict: Dictionary mapping "schema.table" to their DDL schemas
    """
    logger.info("Starting schema-independent extraction for all database tables")
    
    # Create connection
    try:
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT
        )
        logger.info(f"Connected to database {config.DB_NAME} at {config.DB_HOST}:{config.DB_PORT}")
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return {}
    
    try:
        # Get all schemas
        schemas = get_all_schemas(conn)
        logger.info(f"Found {len(schemas)} schemas: {schemas}")
        
        # Dictionary to store all table schemas
        all_schemas = {}
        total_tables = 0
        
        # Extract schema for each schema and its tables
        for schema_name in schemas:
            logger.info(f"Processing schema: {schema_name}")
            
            # Get all tables in this schema
            tables = get_all_tables(schema_name, conn)
            logger.info(f"Found {len(tables)} tables in schema {schema_name}")
            
            # Extract schema for each table
            for table in tables:
                table_key = f"{schema_name}.{table}"
                logger.info(f"Extracting schema for: {table_key}")
                
                ddl = get_table_schema(schema_name, table, conn)
                if ddl:
                    all_schemas[table_key] = ddl
                    logger.info(f"Successfully extracted schema for {table_key}")
                    total_tables += 1
                else:
                    logger.warning(f"Failed to extract schema for {table_key}")
                    all_schemas[table_key] = f"Error: Failed to extract schema for {table_key}"
        
        logger.info(f"Total tables processed: {total_tables}")
        
        # Save to CSV
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "data"
        os.makedirs(data_dir, exist_ok=True)
        
        output_path = data_dir / "table_schema.csv"
        df = pd.DataFrame({
            'table_name': list(all_schemas.keys()),
            'DDL': list(all_schemas.values())
        })
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(all_schemas)} table schemas to {output_path}")
        
        return all_schemas
        
    except Exception as e:
        logger.error(f"Error in schema extraction: {e}")
        return {}
    finally:
        conn.close()
        logger.info("Database connection closed")

