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
from sklearn.metrics import (classification_report, accuracy_score,
                             f1_score, precision_score, recall_score,
                             matthews_corrcoef, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

# 内存优化工具
import gc
from keras import backend as K

# 1. Spark资源配置优化
spark = SparkSession.builder \
    .appName("Text_Classification") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# 2. 使用轻量级模型
MODEL_NAME = "distilbert-base-uncased"
MODEL_SAVE_PATH = "/tmp/text_classification_model"  # 模型保存路径


def load_data(file_path):
    """加载文本分类数据集"""
    try:
        df = spark.read.csv(file_path, sep="\t", header=False, inferSchema=True, quote='"')

        # 检查列数并确定文本和标签列的位置
        num_cols = len(df.columns)
        if num_cols == 4:
            # 格式: [_, label, _, text]
            return df.select(
                col("_c3").alias("text"),
                col("_c1").cast("int").alias("label")
            )
        elif num_cols == 5:
            # 格式: [_, label, _, text, _]
            return df.select(
                col("_c3").alias("text"),
                col("_c1").cast("int").alias("label")
            )
        else:
            raise ValueError(f"无法识别的数据格式，列数: {num_cols}")

    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        spark.stop()
        exit(1)


def create_lightweight_preprocessing_pipeline():
    """创建轻量级特征工程管道"""
    return Pipeline(stages=[
        Tokenizer(inputCol="text", outputCol="words"),
        StopWordsRemover(inputCol="words", outputCol="filtered_words")
    ])


class DistilBERTWrapper:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self.model = None

    def encode_text(self, text, max_length=64):
        """编码文本（限制序列长度以节省内存）"""
        return self.tokenizer(
            text.tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="tf"
        )

    def init_model(self, class_weights=None):
        """初始化分类模型"""
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2  # 二分类任务
        )

        # 使用加权交叉熵损失处理类别不平衡
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        if class_weights:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(2e-5),
            loss=loss,
            metrics=['accuracy']
        )

        self.class_weights = class_weights

    def save_model(self, path):
        """保存模型"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def load_distilbert_model(path):
    """加载DistilBERT模型"""
    tokenizer = DistilBertTokenizer.from_pretrained(path)
    model = TFDistilBertForSequenceClassification.from_pretrained(path)
    return tokenizer, model


def calculate_metrics(y_true, y_pred):
    """计算多种评估指标"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

    # 计算混淆矩阵各项
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp
    })

    return metrics


def plot_results(y_true, y_pred, metrics):
    """绘制结果可视化图表"""
    plt.figure(figsize=(15, 10))

    # 混淆矩阵
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # 指标条形图
    plt.subplot(2, 2, 2)
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'mcc']
    metric_values = [metrics[k] for k in metric_names]
    plt.bar(metric_names, metric_values)
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1)

    # 类别分布对比
    plt.subplot(2, 2, 3)
    unique, counts = np.unique(y_true, return_counts=True)
    plt.bar(unique - 0.1, counts, width=0.2, label='True')
    unique, counts = np.unique(y_pred, return_counts=True)
    plt.bar(unique + 0.1, counts, width=0.2, label='Predicted')
    plt.xticks([0, 1])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Class Distribution Comparison')

    # 指标表格
    plt.subplot(2, 2, 4)
    cell_text = [
        [f"{metrics['accuracy']:.4f}"],
        [f"{metrics['precision']:.4f}"],
        [f"{metrics['recall']:.4f}"],
        [f"{metrics['f1']:.4f}"],
        [f"{metrics['mcc']:.4f}"]
    ]
    plt.table(cellText=cell_text,
              rowLabels=['Accuracy', 'Precision', 'Recall', 'F1', 'MCC'],
              colLabels=['Value'],
              loc='center')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("/tmp/cola_results.png")
    plt.close()


def plot_training_history(history):
    """绘制训练历史图表"""
    plt.figure(figsize=(12, 5))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
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
    plt.savefig("/tmp/cola_training_history.png")
    plt.close()


def main():
    # 数据加载
    data_dir = "file:///datas/glue/CoLA"
    train_df = load_data(f"{data_dir}/train.tsv")
    dev_df = load_data(f"{data_dir}/dev.tsv")

    # 数据预览和类别权重计算
    label_counts = train_df.groupBy("label").count().collect()
    class_dist = {row['label']: row['count'] for row in label_counts}
    print("\n训练集统计:")
    print(f"类别分布: 0={class_dist.get(0, 0)}, 1={class_dist.get(1, 0)}")

    labels = train_df.select("label").toPandas()["label"]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(enumerate(class_weights))
    print(f"自动计算的类别权重: {class_weights}")

    # 轻量级预处理
    pipeline = create_lightweight_preprocessing_pipeline()
    preprocessor = pipeline.fit(train_df)
    train_processed = preprocessor.transform(train_df)
    dev_processed = preprocessor.transform(dev_df)

    # 采样训练数据以节省内存 (可根据资源调整采样比例)
    train_sample = train_processed.select("label", "text") \
        .sample(0.8, seed=42).toPandas()

    # 初始化并训练模型
    bert = DistilBERTWrapper()
    train_encodings = bert.encode_text(
        train_sample['text'],
        max_length=64
    )
    train_labels = train_sample['label'].values

    bert.init_model(class_weights=class_weights)

    print("\n===== 训练配置 ======")
    print(f"模型: DistilBERT (分类版)")
    print(f"训练样本数: {len(train_sample)}")
    print(f"类别权重: {class_weights}")
    print(f"序列最大长度: 64")
    print(f"批次大小: 8 (根据内存调整)")
    print("====================\n")

    # 自定义加权损失函数
    def weighted_loss(y_true, y_pred):
        weights = tf.gather(list(class_weights.values()), tf.cast(y_true, tf.int32))
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True)
        return tf.reduce_mean(loss * weights)

    # 重新编译模型以使用加权损失
    bert.model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-5),
        loss=weighted_loss,
        metrics=['accuracy']
    )

    # 训练模型
    history = bert.model.fit(
        dict(train_encodings),
        train_labels,
        epochs=3,
        batch_size=8,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=2),
            tf.keras.callbacks.ModelCheckpoint(
                '/tmp/classification_checkpoint',
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
    @pandas_udf("int")
    def predict_udf(text: pd.Series) -> pd.Series:
        model_path = os.path.join(SparkFiles.getRootDirectory(), os.path.basename(MODEL_SAVE_PATH))
        tokenizer, model = load_distilbert_model(model_path)

        batch_size = 8  # 根据executor内存调整
        preds = []
        for i in range(0, len(text), batch_size):
            batch_text = text.iloc[i:i + batch_size]
            inputs = tokenizer(
                batch_text.tolist(),
                truncation=True,
                padding='max_length',
                max_length=64,
                return_tensors="tf"
            )
            batch_logits = model(inputs).logits
            batch_preds = tf.argmax(batch_logits, axis=1).numpy()
            preds.extend(batch_preds)

            # 清理内存
            K.clear_session()
            gc.collect()

        return pd.Series(preds)

    # 验证集评估
    print("\n正在评估验证集...")
    predictions = dev_processed.withColumn("pred", predict_udf("text"))
    y_true = predictions.select("label").toPandas()["label"]
    y_pred = predictions.select("pred").toPandas()["pred"]

    # 计算多种评估指标
    metrics = calculate_metrics(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print("\n验证集评估结果:")
    print(f"准确率(Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率(Precision): {metrics['precision']:.4f}")
    print(f"召回率(Recall): {metrics['recall']:.4f}")
    print(f"F1分数(F1-score): {metrics['f1']:.4f}")
    print(f"马修斯相关系数(MCC): {metrics['mcc']:.4f}")
    print("\n混淆矩阵:")
    print(f"真阴性(TN): {metrics['true_negative']} | 假阳性(FP): {metrics['false_positive']}")
    print(f"假阴性(FN): {metrics['false_negative']} | 真阳性(TP): {metrics['true_positive']}")
    print("\n详细分类报告:")
    print(report)

    # 结果可视化
    plot_results(y_true, y_pred, metrics)
    print("结果可视化图已保存至: /tmp/cola_results.png")

    # 训练历史可视化
    plot_training_history(history)
    print("训练历史图已保存至: /tmp/cola_training_history.png")

    # 示例预测对比
    sample_results = predictions.limit(5).toPandas()
    print("\n预测示例对比:")
    for _, row in sample_results.iterrows():
        print(f"\n文本: {row['text'][:50]}...")
        print(f"真实标签: {row['label']}")
        print(f"预测标签: {row['pred']}")

    # 资源释放
    spark.stop()
    K.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()