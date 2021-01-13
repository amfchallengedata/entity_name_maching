from pyspark.sql.functions import *
from pyspark.sql.window import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.linalg import Vectors

def clean_txt(df, col_text, col_clean):
    df = df.withColumn(col_clean, lower(col(col_text)))
    df = df.withColumn(col_clean, regexp_replace(col(col_clean), "-|,", " "))
    df = df.withColumn(col_clean, regexp_replace(col(col_clean), " +", " "))
    df = df.withColumn(col_clean, regexp_replace(col(col_clean), "[éèêë]", "e"))
    df = df.withColumn(col_clean, regexp_replace(col(col_clean), "[àâä]", "a"))
    df = df.withColumn(col_clean, regexp_replace(col(col_clean), "[ôö]", "o"))
    df = df.withColumn(col_clean, regexp_replace(col(col_clean), "[îï]", "i"))
    return df
    

euclidean_distance = udf(lambda v1, v2:  float(Vectors.squared_distance(v1, v2)), FloatType())

def find_LEI(df_base, df_search):
    """Find the mostly liked LEI based on approximate name
    
    Arguments:
        df_base : DataFrame(name:String, lei:String)
        df_search : DataFrame(name:String)
    
    Return:
        DataFrame(lei:String)
    """
    # text preprocessing 
    df_base = clean_txt(df_base, "name", "clean_text")
    df_search = clean_txt(df_search, "name", "clean_text")
    # TF-IDF pipeline
    pipeline = Pipeline().setStages([
        Tokenizer().setInputCol("clean_text").setOutputCol("words"),
        HashingTF().setInputCol("words").setOutputCol("TF"),
        IDF().setInputCol("TF").setOutputCol("features")
    ])
    model = pipeline.fit(df_base)
    df_search = model.transform(df_search)
    df_base = model.transform(df_base)
    df_search = df_search.withColumn("_id", monotonically_increasing_id())
    # Search for closest name
    df_res = (df_search.alias("A").crossJoin(df_base.alias("B"))
        # search only when the 3 first charater are the same 
        .withColumn("t1", substring(col("A.clean_text"), 1, 3))
        .withColumn("t2", substring(col("B.clean_text"), 1, 3))
        .where(col("t1") == col("t2"))
        # compute euclidean distance
        .withColumn("dist", euclidean_distance(col("A.features"), col("B.features")))
        # find smallest distance
        .withColumn("row_number", row_number().over(Window.partitionBy("A._id").orderBy(col("dist"))))
        .where(col("row_number") == 1)
    )
    return df_res.select("lei")