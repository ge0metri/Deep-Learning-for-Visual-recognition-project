# Copyright 2021 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for dasalc.py."""

import os

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized

import tensorflow as tf

from google.protobuf import text_format
from tensorflow_ranking.research import dasalc
from tensorflow_serving.apis import input_pb2

FLAGS = flags.FLAGS

ELWC = text_format.Parse(
    """
    context {
    }
    examples {
      features {
        feature {
          key: "custom_features_1"
          value { float_list { value: 1.0 } }
        }
        feature {
          key: "custom_features_2"
          value { float_list { value: 1.5 } }
        }
        feature {
          key: "utility"
          value { int64_list { value: 1 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "custom_features_1"
          value { float_list { value: 1.0 } }
        }
        feature {
          key: "custom_features_3"
          value { float_list { value: 2.1 } }
        }
        feature {
          key: "utility"
          value { int64_list { value: 0 } }
        }
      }
    }""", input_pb2.ExampleListWithContext())


def _write_tfrecord_files(path):
  elwc_list = [ELWC.SerializeToString()] * 10
  if tf.io.gfile.exists(path):
    tf.io.gfile.remove(path)

  with tf.io.TFRecordWriter(path) as writer:
    for elwc in elwc_list:
      writer.write(elwc)


class TFRankingDASALCTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(TFRankingDASALCTest, self).setUp()
    tf.compat.v1.reset_default_graph()

    # Prepares model directory, and train and eval data.
    self._base_model_dir = tf.compat.v1.test.get_temp_dir() + "/model/"
    tf.io.gfile.makedirs(self._base_model_dir)
    self._data_file = os.path.join(self._base_model_dir, "elwc.tfrecord")
    _write_tfrecord_files(self._data_file)

  def tearDown(self):
    super(TFRankingDASALCTest, self).tearDown()
    if self._base_model_dir:
      tf.io.gfile.rmtree(self._base_model_dir)
    self._base_model_dir = None

  def test_train_and_eval(self):
    self._model_dir = self._base_model_dir + "/" + "dasalc"
    with flagsaver.flagsaver(
        train_input_pattern=self._data_file,
        eval_input_pattern=self._data_file,
        model_dir=self._model_dir,
        batch_size=16,
        hidden_layer_dims=64,
        num_attention_layers=1,
        num_attention_heads=1,
        list_size=100,
        num_features=3,
        num_train_steps=10):
      dasalc.train_and_eval()


if __name__ == "__main__":
  tf.test.main()
