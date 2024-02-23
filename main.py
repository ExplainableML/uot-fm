from absl import app
from absl import flags
import blobfile as bf
from ml_collections.config_flags import config_flags
import logging
import os

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config")
flags.DEFINE_string("workdir", "runs", "Work directory.")
flags.DEFINE_enum("mode", "train", ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("devices", None, "CUDA devices to use.")
flags.mark_flags_as_required(["config"])


def main(argv):
    # Create the working directory
    bf.makedirs(f"{FLAGS.workdir}/logs")
    logger = logging.getLogger()
    file_stream = open(f"{FLAGS.workdir}/logs/{FLAGS.config.name}.txt", "w")
    handler = logging.StreamHandler(file_stream)
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel("INFO")
    if FLAGS.devices is not None:
        logging.info(f"Using CUDA devices {FLAGS.devices}")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.devices

    FLAGS.config.wandb_key = os.environ.get("WANDB_API_KEY", FLAGS.config.wandb_key)
    if FLAGS.mode == "train":
        import train

        train.train(FLAGS.config, FLAGS.workdir)
        file_stream.close()
    else:
        import evaluate

        evaluate.evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    app.run(main)
