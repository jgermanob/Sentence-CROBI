import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras import layers
from tensorflow.keras.initializers import TruncatedNormal
import tensorflow.keras.backend as K
from transformers import TFRobertaModel, TFBertModel
from tensorflow.keras import Input
from tensorflow.keras.models import Model

class Mean_pooling(layers.Layer):
    def __init__(self):
        super(Mean_pooling, self).__init__()
    
    def call(self, tensors):
        token_embeddings, attention_mask = tensors
        input_mask_expanded = tf.cast(K.expand_dims(attention_mask,axis=-1), dtype=tf.float16)
        return tf.reduce_sum(token_embeddings * input_mask_expanded,1)/tf.clip_by_value(tf.math.reduce_sum(input_mask_expanded,1), clip_value_min=1e-9, clip_value_max=1e9)

class Max_pooling(layers.Layer):
    def __init__(self):
        super(Max_pooling, self).__init__()

    def call(self, tensors):
        token_embeddings, attention_mask = tensors
        input_mask_expanded = tf.cast(K.expand_dims(attention_mask,axis=-1), dtype=tf.float16)
        input_mask_expanded = tf.tile(input_mask_expanded,multiples=(1,1,768))
        token_embeddings = tf.where(tf.math.not_equal(input_mask_expanded,tf.constant([0.0], dtype=tf.float16)),token_embeddings,[-1e9])
        return tf.math.reduce_max(token_embeddings,1)
    
class Euclidean_distance(layers.Layer):
    def __init__(self):
        super(Euclidean_distance, self).__init__()
    
    def call(self, vectors):
        (featsA, featsB) = vectors
        # compute the sum of squared distances between the vectors
        sumSquared = K.sum(K.square(featsA - featsB), axis=1,keepdims=True)
        # return the euclidean distance between the vectors
        return K.sqrt(K.maximum(sumSquared, K.epsilon()))

class Roberta_classifier(layers.Layer):
    def __init__(self, classes):
        super(Roberta_classifier, self).__init__()
        self.dropout = layers.Dropout(0.1, name='dropout_classifier')
        self.dense_1 = layers.Dense(1793, activation='tanh', kernel_initializer=TruncatedNormal(stddev=0.02))
        self.dense_2 = layers.Dense(classes, kernel_initializer=TruncatedNormal(stddev=0.02))
    
    def call(self, x):
        x = self.dropout(x)
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        return x


def build_Sentence_CROBI_model(pooling_strategy='mean', classes=2, crossencoder_checkpoint='roberta-large', biencoder_checkpoint='bert-base-uncased'):
    # Definición de las entradas #
    input_id = Input(shape=(128,), name='input_id', dtype=tf.int32)
    attention_mask = Input(shape=(128,), name='attention_mask', dtype=tf.int32)
    token_type_ids = Input(shape=(128,), name='token_type_ids', dtype=tf.int32)
    
    left_input_id = Input(shape=(35,), name='left_input_id', dtype=tf.int32)
    left_attention_mask = Input(shape=(35,), name='left_attention_mask', dtype=tf.int32)
    left_token_type_ids = Input(shape=(35,), name='left_token_type_ids', dtype=tf.int32)
    
    right_input_id = Input(shape=(35,), name='right_input_id', dtype=tf.int32)
    right_attention_mask = Input(shape=(35,), name='right_attention_mask', dtype=tf.int32)
    right_token_type_ids = Input(shape=(35,), name='right_token_type_ids', dtype=tf.int32)
    
    # Definición de la arquitectura #
    roberta = TFRobertaModel.from_pretrained(crossencoder_checkpoint,
                                       output_attentions = False, 
                                       output_hidden_states = False,
                                       return_dict=True,
                                       use_cache=True,
                                       name='cross_encoder')

    siamese_bert = TFBertModel.from_pretrained(biencoder_checkpoint,
                                       output_attentions = False, 
                                       output_hidden_states = False,
                                       return_dict=True,
                                       use_cache=True,
                                       from_pt=True,
                                       name='bi_encoder')
    
    if pooling_strategy == 'mean':
        pooling = Mean_pooling()
    else:
        pooling = Max_pooling()
    
    classifier = Roberta_classifier(classes=classes)

    # Forward pass #
    s_token = roberta(input_id, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
    s_token = s_token[:,0,:]
    s_token = layers.Dropout(0.3)(s_token)

    left_output = siamese_bert(left_input_id, attention_mask=left_attention_mask, token_type_ids=left_token_type_ids)[0]
    left_vector = pooling([left_output, left_attention_mask])

    right_output = siamese_bert(right_input_id, attention_mask=right_attention_mask, token_type_ids=right_token_type_ids)[0]
    right_vector = pooling([right_output, right_attention_mask])

    semantic_vector = layers.Concatenate(name='concatenate_siamese_output')([left_vector, right_vector])
    semantic_vector = layers.Dropout(0.3)(semantic_vector)

    semantic_distance = Euclidean_distance()([right_vector, left_vector])
    concatted = layers.Concatenate(name='global_vector')([s_token, semantic_vector, semantic_distance])

    predictions = classifier(concatted)

    # Definición del modelo
    model = Model([input_id, attention_mask, token_type_ids,
                   left_input_id, left_attention_mask, left_token_type_ids,
                   right_input_id, right_attention_mask, right_token_type_ids], 
                  predictions)

    return model