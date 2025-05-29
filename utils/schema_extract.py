"""Schema extraction utilities for the RAG application."""

import os
import logging
import pandas as pd
import psycopg2
from pathlib import Path
from config.settings import DB_CONFIG

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
            tc.table_schema = %s
            AND tc.table_name = %s
            AND tc.constraint_type = 'PRIMARY KEY';
        """

        # Execute the queries
        cursor.execute(column_query, (schema_name, table_name))
        columns = cursor.fetchall()

        cursor.execute(primary_key_query, (schema_name, table_name))
        primary_keys = [row[0] for row in cursor.fetchall()]

        # Start building the CREATE TABLE statement
        create_table_query = f"CREATE TABLE {schema_name}.{table_name} (\n"

        # Add columns to the CREATE TABLE statement
        column_definitions = []
        for column in columns:
            column_name = column[0]
            data_type = column[1]
            char_length = column[2]
            is_nullable = column[3]

            # Handle data type with length (e.g., varchar(50))
            if char_length:
                data_type = f"{data_type}({char_length})"

            # Handle NOT NULL constraint
            null_constraint = "NOT NULL" if is_nullable == "NO" else "NULL"

            column_definitions.append(f"\t{column_name} {data_type} {null_constraint}")

        # Add columns to the final query
        create_table_query += ",\n".join(column_definitions)

        # Add primary key constraint
        if primary_keys:
            primary_keys_str = ", ".join(primary_keys)
            create_table_query += f",\n\tCONSTRAINT {table_name}_id PRIMARY KEY ({primary_keys_str})"

        # Close the table
        create_table_query += "\n);"

        return create_table_query

    except Exception as e:
        logger.error(f"Error extracting schema for {table_name}: {e}")
        return None
    finally:
        cursor.close()

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
    Extract schemas for all tables in the database.
    
    Returns:
        dict: Dictionary mapping table names to their DDL schemas
    """
    # Get database configuration
    db_schema = DB_CONFIG.get("db_schema", "customersetup")
    logger.info(f"Using database schema: {db_schema}")
    
    # Create connection
    try:
        conn = psycopg2.connect(
            dbname=DB_CONFIG.get("db_name"),
            user=DB_CONFIG.get("db_user"),
            password=DB_CONFIG.get("db_password"),
            host=DB_CONFIG.get("db_host"),
            port=DB_CONFIG.get("db_port")
        )
        logger.info(f"Connected to database {DB_CONFIG.get('db_name')} at {DB_CONFIG.get('db_host')}:{DB_CONFIG.get('db_port')}")
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return {}
    
    try:
        # Get all tables
        tables = get_all_tables(db_schema, conn)
        logger.info(f"Found {len(tables)} tables in schema {db_schema}")
        
        # Dictionary to store schemas
        schemas = {}
        
        # Extract schema for each table
        for table in tables:
            logger.info(f"Extracting schema for table: {table}")
            ddl = get_table_schema(db_schema, table, conn)
            if ddl:
                schemas[table] = ddl
                logger.info(f"Successfully extracted schema for {table}")
            else:
                logger.warning(f"Failed to extract schema for {table}")
                schemas[table] = f"Error: Failed to extract schema for {table}"
        
        # Save to CSV
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "data"
        os.makedirs(data_dir, exist_ok=True)
        
        output_path = data_dir / "table_schema.csv"
        df = pd.DataFrame({
            'table_name': list(schemas.keys()),
            'DDL': list(schemas.values())
        })
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(schemas)} table schemas to {output_path}")
        
        return schemas
        
    except Exception as e:
        logger.error(f"Error in schema extraction: {e}")
        return {}
    finally:
        conn.close()
        logger.info("Database connection closed")

