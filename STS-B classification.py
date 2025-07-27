import os
from pyspark import SparkFiles
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志输出
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score  # 新增r2_score

# 内存优化工具
import gc
from keras import backend as K

# 1. Spark资源配置优化
spark = SparkSession.builder \
    .appName("STS-B_Similarity_Analysis") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# 2. 使用轻量级模型
MODEL_NAME = "distilbert-base-uncased"
MODEL_SAVE_PATH = "/tmp/stsb_model"  # 模型保存路径

def load_stsb_data(file_path, has_labels=True):
    """加载STS-B数据集"""
    try:
        df = spark.read.csv(file_path, sep="\t", header=True, inferSchema=True, quote='"')
        if has_labels:
            return df.select(
                col("sentence1").alias("text1"),
                col("sentence2").alias("text2"),
                col("score").cast("float").alias("label"))
        else:
            return df.select(
                col("index").alias("index"),
                col("sentence1").alias("text1"),
                col("sentence2").alias("text2"))
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        spark.stop()
        exit(1)

def create_lightweight_preprocessing_pipeline():
    """创建轻量级特征工程管道"""
    return Pipeline(stages=[
        Tokenizer(inputCol="text1", outputCol="words1"),
        Tokenizer(inputCol="text2", outputCol="words2"),
        StopWordsRemover(inputCol="words1", outputCol="filtered_words1"),
        StopWordsRemover(inputCol="words2", outputCol="filtered_words2")
    ])

class DistilBERTWrapper:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self.model = None

    def encode_text_pairs(self, text1, text2, max_length=64):
        """编码文本对（限制序列长度以节省内存）"""
        return self.tokenizer(
            text1.tolist(),
            text2.tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="tf"
        )

    def init_regression_model(self):
        """初始化回归模型"""
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1  # 回归任务输出1个值
        )
        # 使用更小的学习率和MSE损失
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['mse']
        )

    def save_model(self, path):
        """保存模型"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

def load_distilbert_model(path):
    """加载DistilBERT模型"""
    tokenizer = DistilBertTokenizer.from_pretrained(path)
    model = TFDistilBertForSequenceClassification.from_pretrained(path)
    return tokenizer, model

def plot_results(y_true, y_pred):
    """绘制结果可视化图表"""
    plt.figure(figsize=(12, 5))

    # 真实值 vs 预测值散点图
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 5], [0, 5], 'r--')
    plt.xlabel('True Score')
    plt.ylabel('Predicted Score')
    plt.title('True vs Predicted Scores')

    # 误差分布直方图
    plt.subplot(1, 2, 2)
    errors = y_pred - y_true
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.title('Error Distribution')

    plt.tight_layout()
    plt.savefig("/tmp/stsb_results.png")
    plt.close()

def plot_training_history(history):
    """绘制训练历史图表"""
    plt.figure(figsize=(12, 5))

    # MSE曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mse'], label='Train MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title('Model MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig("/tmp/stsb_training_history.png")
    plt.close()

def main():
    # 数据加载
    data_dir = "file:///datas/glue/STS-B"
    train_df = load_stsb_data(f"{data_dir}/train.tsv")
    dev_df = load_stsb_data(f"{data_dir}/dev.tsv")
    test_df = load_stsb_data(f"{data_dir}/test.tsv", has_labels=False)

    # 数据预览
    print("\n训练集统计:")
    train_df.select("label").describe().show()

    # 轻量级预处理
    pipeline = create_lightweight_preprocessing_pipeline()
    preprocessor = pipeline.fit(train_df)
    train_processed = preprocessor.transform(train_df)
    dev_processed = preprocessor.transform(dev_df)

    # 采样训练数据以节省内存 (可根据资源调整采样比例)
    train_sample = train_processed.select("label", "text1", "text2") \
        .sample(0.8, seed=42).toPandas()

    # 初始化并训练模型
    bert = DistilBERTWrapper()
    train_encodings = bert.encode_text_pairs(
        train_sample['text1'],
        train_sample['text2'],
        max_length=64
    )
    train_labels = train_sample['label'].values

    bert.init_regression_model()

    print("\n===== 训练配置 ======")
    print(f"模型: DistilBERT (回归版)")
    print(f"训练样本数: {len(train_sample)}")
    print(f"序列最大长度: 64")
    print(f"批次大小: 8 (根据内存调整)")
    print("====================\n")

    # 训练模型 (减少epoch数以节省时间)
    history = bert.model.fit(
        dict(train_encodings),
        train_labels,
        epochs=3,
        batch_size=4,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=2),
            tf.keras.callbacks.ModelCheckpoint(
                '/tmp/stsb_checkpoint',
                save_best_only=True
            )
        ]
    )

    # 保存模型
    bert.save_model(MODEL_SAVE_PATH)
    print(f"模型已保存至: {MODEL_SAVE_PATH}")

    # 将模型分发到集群
    spark.sparkContext.addFile(MODEL_SAVE_PATH, recursive=True)

    # 定义预测UDF (分批处理以避免OOM)
    @pandas_udf("float")
    def predict_udf(text1: pd.Series, text2: pd.Series) -> pd.Series:
        model_path = os.path.join(SparkFiles.getRootDirectory(), os.path.basename(MODEL_SAVE_PATH))
        tokenizer, model = load_distilbert_model(model_path)

        batch_size = 8  # 根据executor内存调整
        preds = []
        for i in range(0, len(text1), batch_size):
            batch_text1 = text1.iloc[i:i + batch_size]
            batch_text2 = text2.iloc[i:i + batch_size]
            inputs = tokenizer(
                batch_text1.tolist(),
                batch_text2.tolist(),
                truncation=True,
                padding='max_length',
                max_length=64,
                return_tensors="tf"
            )
            batch_preds = model(inputs).logits.numpy().flatten()
            preds.extend(batch_preds)

            # 清理内存
            K.clear_session()
            gc.collect()

        return pd.Series(preds)

    # 验证集评估
    print("\n正在评估验证集...")
    predictions = dev_processed.withColumn("pred", predict_udf("text1", "text2"))
    y_true = predictions.select("label").toPandas()["label"]
    y_pred = predictions.select("pred").toPandas()["pred"]

    # 计算评估指标
    pearson = pearsonr(y_true, y_pred)[0]
    spearman = spearmanr(y_true, y_pred)[0]
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)  # 新增R²计算

    print(f"\n验证集评估结果:")
    print(f"Pearson相关系数: {pearson:.4f}")
    print(f"Spearman相关系数: {spearman:.4f}")
    print(f"均方误差(MSE): {mse:.4f}")
    print(f"R平方(R²): {r2:.4f}")  # 新增R²输出

    # 结果可视化
    plot_results(y_true, y_pred)
    print("结果可视化图已保存至: /tmp/stsb_results.png")

    # 训练历史可视化
    plot_training_history(history)
    print("训练历史图已保存至: /tmp/stsb_training_history.png")

    # 测试集预测
    print("\n正在预测测试集...")
    test_predictions = test_df.withColumn("prediction", predict_udf("text1", "text2"))

    # 保存结果
    output_dir = "file:///tmp/stsb_predictions"
    test_predictions.select("index", "prediction") \
        .write.mode("overwrite").csv(output_dir, header=True)
    print(f"测试集预测结果已保存至: {output_dir}")

    # 示例预测对比
    sample_results = predictions.limit(5).toPandas()
    print("\n预测示例对比:")
    for _, row in sample_results.iterrows():
        print(f"\n句子1: {row['text1'][:30]}...")
        print(f"句子2: {row['text2'][:30]}...")
        print(f"真实分数: {row['label']:.2f}")
        print(f"预测分数: {row['pred']:.2f}")

    # 资源释放
    spark.stop()
    K.clear_session()
    gc.collect()

if __name__ == "__main__":
    main()