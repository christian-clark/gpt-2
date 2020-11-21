#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder, surprisal

BATCH_SIZE = 1

def per_subword_surp(
    text, 
    model_name='124M',
    models_dir='models',
):
    """
    Find per-subword surprisal for a provided text
    :text : path to input text file
    :model_name=124M : String, which model to use
    :models_dir : path to parent folder containing model subfolders
    (i.e. contains the <model_name> folder)
    """

    text = open(text).read().strip()

    enc = encoder.get_encoder(model_name, models_dir)
    enc_text = enc.encode(text)
    word_pieces = [enc.decode([x]) for x in enc_text]

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))


    with tf.Session(graph=tf.Graph()) as sess:

        output = surprisal.get_per_word_surprisal(
            corpus=enc_text, hparams=hparams,
            encoder=enc
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        # out has dimension batch_size x num_subwords
        # containing per-subword surprisals
        out = sess.run(output)
        print('word_piece\tencoding\tsurprisal')
        for i in range(BATCH_SIZE):
            surps = out[i]
            assert len(word_pieces) == len(enc_text) == len(surps)
            for j in range(len(surps)):
                print('"{}"\t{}\t{}'.format(word_pieces[j], enc_text[j], surps[j]))


if __name__ == '__main__':
    fire.Fire(per_subword_surp)
