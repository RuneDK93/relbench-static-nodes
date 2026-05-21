import duckdb
import pandas as pd

from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
    mae,
    r2,
    rmse,
    roc_auc,
)


class UserItemPurchaseTask(RecommendationTask):
    r"""Predict the list of articles each customer will purchase in the next seven
    days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "article_id"
    dst_entity_table = "article"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=7)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 12

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                transactions.customer_id,
                LIST(DISTINCT transactions.article_id) AS article_id
            FROM
                timestamp_df t
            LEFT JOIN
                transactions
            ON
                transactions.t_dat > t.timestamp AND
                transactions.t_dat <= t.timestamp + INTERVAL '{self.timedelta} days'
            GROUP BY
                t.timestamp,
                transactions.customer_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )


class UserChurnTask(EntityTask):
    r"""Predict the churn for a customer (no transactions) in the next week."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=7)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                timestamp,
                customer_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM transactions
                        WHERE
                            transactions.customer_id = customer.customer_id AND
                            t_dat > timestamp AND
                            t_dat <= timestamp + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS churn
            FROM
                timestamp_df,
                customer,
            WHERE
                EXISTS (
                    SELECT 1
                    FROM transactions
                    WHERE
                        transactions.customer_id = customer.customer_id AND
                        t_dat > timestamp - INTERVAL '{self.timedelta}' AND
                        t_dat <= timestamp
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class ItemSalesTask(EntityTask):
    r"""Predict the total sales for an article (the sum of prices of the associated
    transactions) in the next week."""

    task_type = TaskType.REGRESSION
    entity_col = "article_id"
    entity_table = "article"
    time_col = "timestamp"
    target_col = "sales"
    timedelta = pd.Timedelta(days=7)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        article = db.table_dict["article"].df

        df = duckdb.sql(
            f"""
            SELECT
                timestamp,
                article_id,
                sales
            FROM
                timestamp_df,
                article,
                (
                    SELECT
                        COALESCE(SUM(price), 0) as sales
                    FROM
                        transactions,
                    WHERE
                        transactions.article_id = article.article_id AND
                        t_dat > timestamp AND
                        t_dat <= timestamp + INTERVAL '{self.timedelta}'
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"article_id": "article"},
            pkey_col=None,
            time_col="timestamp",
        )



class CustomerAgeNthTransactionTask(EntityTask):
    r"""
    Predict the age of a customer after their nth recorded transaction timestamp,
    using only graph information available up to that point.

    This is a static node classification task:
    - The target is a static property of the customer node.
    - Each customer with n transactions is included once, aligned to their nth distinct transaction timestamp.
    """

    task_type = TaskType.REGRESSION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "nth_time"   # The selected time for modelling
    target_col = "age"
    metrics = [r2, mae, rmse]

   # dummy timedelta and num_eval_timestamps for BaseTask checks. Not used for windowing here
    timedelta = pd.Timedelta(days=7)
    num_eval_timestamps = 1    

    # Special attribute for time-independent node property tasks. 
    # This list specifies which features to remove from the INPUT graph to the task.
    # This should be the target feature, or any features that directly map to the target feature.    
    remove_columns = ['age']

    # Flag that this is a special type of time-independent node property task
    time_independent_node_task = True    

    # Special attribute for time-independent node property tasks. 
    # This number indicates how many interactions (here transactions) the target nodes should have at prediction time.
    n = 30  # transaction number to align to
    interaction_table = 'transactions'
    interaction_table_time_col = 't_dat'        

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:

        """
        Construct a table containing the nth interaction timestamp and the corresponding target value for each entity.

        Steps performed:

        distinct_interactions:
           - Selects all distinct interaction timestamps for each entity from the interaction table.

        entity_interaction_ranks:
           - Assigns a sequential number (interaction_number) to each interaction per entity
             based on the interaction timestamp (earliest first). This allows identifying the nth interaction for each entity.

        nth_interaction:
           - Filters the interactions to keep only the nth interaction timestamp per entity.

        Final SELECT:
           - Joins the nth interaction with the entity table to get the target value.
           - Filters out entities with NULL target values.
           - Orders the resulting table by the interaction timestamp.

        """        

        con = duckdb.connect()
        con.register(self.interaction_table, db.table_dict[self.interaction_table].df)
        con.register(self.entity_table, db.table_dict[self.entity_table].df)        

        query = f"""
            WITH distinct_interactions AS (
                SELECT DISTINCT
                    {self.entity_col},
                    {self.interaction_table_time_col},
                FROM {self.interaction_table}
            ),
            entity_interaction_ranks AS (
                SELECT
                    {self.entity_col},
                    {self.interaction_table_time_col},
                    ROW_NUMBER() OVER (PARTITION BY {self.entity_col} ORDER BY {self.interaction_table_time_col}) AS interaction_number
                FROM distinct_interactions
            ),
            nth_interaction AS (
                SELECT
                    {self.entity_col},
                    {self.interaction_table_time_col} AS {self.time_col}
                FROM entity_interaction_ranks
                WHERE interaction_number = {self.n}
            )
            SELECT
                ni.{self.entity_col},
                ni.{self.time_col},
                et.{self.target_col}
            FROM nth_interaction ni
            JOIN {self.entity_table} et
              ON et.{self.entity_col} = ni.{self.entity_col}
            WHERE et.{self.target_col} IS NOT NULL 
            ORDER BY ni.{self.time_col} ASC
        """

        df = con.execute(query).df()

        # Infer split based on timestamps
        if len(timestamps) > 1 and timestamps[0] > timestamps[1]:
            split = "train"
        elif timestamps[0] == self.dataset.val_timestamp:
            split = "val"
        elif timestamps[0] == self.dataset.test_timestamp:
            split = "test"
        else:
            raise ValueError("Could not infer split from timestamps")

        val_ts = self.dataset.val_timestamp
        test_ts = self.dataset.test_timestamp

        # Then perform the split and return the associated table
        if split == "train":
            df = df[df[self.time_col] < val_ts] 
        elif split == "val":
            df = df[(df[self.time_col] >= val_ts) & (df[self.time_col] < test_ts)]
        elif split == "test":
            df = df[(df[self.time_col] >= test_ts)]

        df = df.reset_index(drop=True)
        
        return Table(
            df=df,
            fkey_col_to_pkey_table={"customer_id": "customer"},
            pkey_col="customer_id",
            time_col=self.time_col,
        )
    


class ArticleIndexNthTransactionTask(EntityTask):
    """
    Predict the index group of an article after its nth recorded transaction,
    using only graph information available up to that point.

    This is a static node classification task:
    - The target is a static property of the article node.
    - Each article having n transactions is included once, aligned to its nth recorded transaction in the "transactions" table. 
    - Timedeltas and prediction windows are not used. Instead we predict on each article node at the timestamp of their nth transaction.
    """
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    entity_col = "article_id" # node identifier
    entity_table = "article" # node table
    time_col = "nth_time"   # The selected time for modelling. Here when the article first appears in a transaction
    target_col = "index_group_no" # target column 

    # dummy timedelta and num_eval_timestamps for BaseTask checks. Not used for windowing here
    timedelta = pd.Timedelta(days=7)
    num_eval_timestamps = 1    

    # Metrics and number of labels
    metrics = [accuracy]
    num_labels = 5 #3 # We are remapping the colors to 3 categories -> Neutral, Colored, Other

    # Special attribute for static node property tasks. 
    # This list specifies which features to remove from the INPUT graph to the task.
    # This should be the target feature, or any features that directly map to the target feature.
    remove_columns = ['index_group_no', # target
                    
                    'index_group_name', # Features that are very correlated with with target
                    'index_name',
                    'index_code',                
                    'product_code',
                    'prod_name',
                    'product_group_name',
                    'product_type_name',
                    'product_type_no',                
                    'department_name',
                    'department_no',  
                    'section_name',
                    'section_no',                
                    'garment_group_name',
                    'garment_group_no',     
                    'detail_desc']  

    # Flag that this is a special type of time-independent node property task
    time_independent_node_task = True                        
    
    # Special attribute for time-independent node property tasks. 
    # This number indicates how many interactions (here transactions) the target nodes should have at prediction time.
    n = 30  # transaction number to align to
    interaction_table = 'transactions'
    interaction_table_time_col = 't_dat'               

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:

        """
        Construct a table containing the nth interaction timestamp and the corresponding target value for each entity.

        Steps performed:

        distinct_interactions:
           - Selects all distinct interaction timestamps for each entity from the interaction table.

        entity_interaction_ranks:
           - Assigns a sequential number (interaction_number) to each interaction per entity
             based on the interaction timestamp (earliest first). This allows identifying the nth interaction for each entity.

        nth_interaction:
           - Filters the interactions to keep only the nth interaction timestamp per entity.

        Final SELECT:
           - Joins the nth interaction with the entity table to get the target value.
           - Filters out entities with NULL target values.
           - Orders the resulting table by the interaction timestamp.

        """        

        con = duckdb.connect()
        con.register(self.interaction_table, db.table_dict[self.interaction_table].df)
        con.register(self.entity_table, db.table_dict[self.entity_table].df)        

        query = f"""
            WITH distinct_interactions AS (
                SELECT DISTINCT
                    {self.entity_col},
                    {self.interaction_table_time_col},
                FROM {self.interaction_table}
            ),
            entity_interaction_ranks AS (
                SELECT
                    {self.entity_col},
                    {self.interaction_table_time_col},
                    ROW_NUMBER() OVER (PARTITION BY {self.entity_col} ORDER BY {self.interaction_table_time_col}) AS interaction_number
                FROM distinct_interactions
            ),
            nth_interaction AS (
                SELECT
                    {self.entity_col},
                    {self.interaction_table_time_col} AS {self.time_col}
                FROM entity_interaction_ranks
                WHERE interaction_number = {self.n}
            )
            SELECT
                ni.{self.entity_col},
                ni.{self.time_col},
                et.{self.target_col}
            FROM nth_interaction ni
            JOIN {self.entity_table} et
              ON et.{self.entity_col} = ni.{self.entity_col}
            WHERE et.{self.target_col} IS NOT NULL 
            ORDER BY ni.{self.time_col} ASC
        """

        df = con.execute(query).df()

        # Mapping categories
        def map_target(code):
            if code == 1:
                return 'Ladieswear'
            elif code == 4:
                return 'Baby/Children'     
            elif code == 2:
                return 'Divided' 
            elif code == 3:
                return 'Menswear'  
            elif code == 26:
                return 'Sport'                                                

        fixed_integer_map = {
            'Ladieswear': 0,    
            'Baby/Children': 1,                      
            'Divided':2,
            'Menswear':3,
            'Sport':4, 
            }

        df["index_group"] = df["index_group_no"].apply(map_target)
        df["index_group_no"] = df["index_group"].map(fixed_integer_map)

        # Infer split based on timestamps
        if len(timestamps) > 1 and timestamps[0] > timestamps[1]:
            split = "train"
        elif timestamps[0] == self.dataset.val_timestamp:
            split = "val"
        elif timestamps[0] == self.dataset.test_timestamp:
            split = "test"
        else:
            raise ValueError("Could not infer split from timestamps")

        val_ts = self.dataset.val_timestamp
        test_ts = self.dataset.test_timestamp

        # Then perform the split and return the associated table
        if split == "train":
            df = df[df[self.time_col] < val_ts] 
        elif split == "val":
            df = df[(df[self.time_col] >= val_ts) & (df[self.time_col] < test_ts)]
        elif split == "test":
            df = df[(df[self.time_col] >= test_ts)]

        df = df.reset_index(drop=True)


        return Table(
            df=df,
            fkey_col_to_pkey_table={"article_id": "article"},
            pkey_col="article_id",
            time_col=self.time_col, # Setting the time column
        )



class ArticleColourNthTransactionTask(EntityTask):
    """
    Predict the static property color of a product after its nth transaction,
    using only graph information available up to that point.

    This is a static node classification task:
    - The target is a static property of the article node.
    - Each article is included once, aligned to its nth recorded transaction in the "transactions" table. 
    - Timedeltas and prediction windows are not used. Instead we predict on each article node at the timestamp of their nth transaction.
    """
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    entity_col = "article_id" # node identifier
    entity_table = "article" # node table
    time_col = "nth_time"   # The selected time for modelling. Here when the article first appears in a transaction
    target_col = "colour_group_code" # target column colour_group_code

    # dummy timedelta and num_eval_timestamps for BaseTask checks. Not used for windowing here
    timedelta = pd.Timedelta(days=7)
    num_eval_timestamps = 1    

    # Metrics and number of labels
    metrics = [accuracy]
    num_labels = 3  # We are remapping the colors to 3 categories -> Neutral, Colored, Other

    # Special attribute for static node property tasks. 
    # This list specifies which features to remove from the INPUT graph to the task.
    # This should be the target feature, or any features that directly map to the target feature.      
    remove_columns = ['colour_group_code', # target
                    
                    'colour_group_name',# Features that are very correlated with with target
                    'perceived_colour_value_name',
                    'perceived_colour_value_id',
                    'perceived_colour_master_name',
                    'perceived_colour_master_id']


    # Flag that this is a special type of time-independent node property task
    time_independent_node_task = True                    
    
    # Special attribute for time-independent node property tasks. 
    # This number indicates how many interactions (here transactions) the target nodes should have at prediction time.
    n = 30  # transaction number to align to
    interaction_table = 'transactions'
    interaction_table_time_col = 't_dat'               

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:

        """
        Construct a table containing the nth interaction timestamp and the corresponding target value for each entity.

        Steps performed:

        distinct_interactions:
           - Selects all distinct interaction timestamps for each entity from the interaction table.

        entity_interaction_ranks:
           - Assigns a sequential number (interaction_number) to each interaction per entity
             based on the interaction timestamp (earliest first). This allows identifying the nth interaction for each entity.

        nth_interaction:
           - Filters the interactions to keep only the nth interaction timestamp per entity.

        Final SELECT:
           - Joins the nth interaction with the entity table to get the target value.
           - Filters out entities with NULL target values.
           - Orders the resulting table by the interaction timestamp.

        """        

        con = duckdb.connect()
        con.register(self.interaction_table, db.table_dict[self.interaction_table].df)
        con.register(self.entity_table, db.table_dict[self.entity_table].df)        

        query = f"""
            WITH distinct_interactions AS (
                SELECT DISTINCT
                    {self.entity_col},
                    {self.interaction_table_time_col},
                FROM {self.interaction_table}
            ),
            entity_interaction_ranks AS (
                SELECT
                    {self.entity_col},
                    {self.interaction_table_time_col},
                    ROW_NUMBER() OVER (PARTITION BY {self.entity_col} ORDER BY {self.interaction_table_time_col}) AS interaction_number
                FROM distinct_interactions
            ),
            nth_interaction AS (
                SELECT
                    {self.entity_col},
                    {self.interaction_table_time_col} AS {self.time_col}
                FROM entity_interaction_ranks
                WHERE interaction_number = {self.n}
            )
            SELECT
                ni.{self.entity_col},
                ni.{self.time_col},
                et.{self.target_col}
            FROM nth_interaction ni
            JOIN {self.entity_table} et
              ON et.{self.entity_col} = ni.{self.entity_col}
            WHERE et.{self.target_col} IS NOT NULL 
            ORDER BY ni.{self.time_col} ASC
        """

        df = con.execute(query).df()

        # Mapping categories
        def map_simplified_colour(code):
            if code in  range(6, 12):
                return 'Neutral'
            elif code in range(12, 100):
                return 'Colored'
            else:
                return 'Other'

        fixed_integer_map = {
            'Neutral': 0,    
            'Colored': 1,                      
            'Other':2 }


        df["simplified_colour_group"] = df["colour_group_code"].apply(map_simplified_colour)
        df["colour_group_code"] = df["simplified_colour_group"].map(fixed_integer_map)

        # Infer split based on timestamps
        if len(timestamps) > 1 and timestamps[0] > timestamps[1]:
            split = "train"
        elif timestamps[0] == self.dataset.val_timestamp:
            split = "val"
        elif timestamps[0] == self.dataset.test_timestamp:
            split = "test"
        else:
            raise ValueError("Could not infer split from timestamps")

        val_ts = self.dataset.val_timestamp
        test_ts = self.dataset.test_timestamp

        # Then perform the split and return the associated table
        if split == "train":
            df = df[df[self.time_col] < val_ts] 
        elif split == "val":
            df = df[(df[self.time_col] >= val_ts) & (df[self.time_col] < test_ts)]
        elif split == "test":
            df = df[(df[self.time_col] >= test_ts)]

        df = df.reset_index(drop=True)


        return Table(
            df=df,
            fkey_col_to_pkey_table={"article_id": "article"},
            pkey_col="article_id",
            time_col=self.time_col, # Setting the time column
        )
    

