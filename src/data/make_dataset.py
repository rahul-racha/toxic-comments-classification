# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv
#import pandas as pd
import BasicPreprocessor as bp

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

def do_basic_preprocess(filePath, interim_filepath):
    trainData = pd.read_csv(filePath)
    print(trainData[['comment_text']].head())
    print('\n\n')

    ref_basic = bp.BasicPreprocessor(trainData[:1], "comment_text")
    tdf = pd.DataFrame({'A' : []})
    ref_basic.perform_operation(bp.Operations.LOWER, tdf, True, True)
    ref_basic.perform_operation(bp.Operations.PUNCTUATION, tdf, True, True)
    ref_basic.perform_operation(bp.Operations.STOPWORDS, tdf, True, True)
#    ref_basic.perform_operation(bp.Operations.CWORDS, tdf, True, True)
#    ref_basic.perform_operation(bp.Operations.RWORDS, tdf, True, True)
#    temp_df = ref_basic.perform_operation(bp.Operations.SPELL, tdf, True, True)
    temp_df = ref_basic.perform_operation(bp.Operations.LEMMA, tdf, True, True)

    temp_df.set_index('id')
    temp_df.to_csv(interim_filepath, sep=',', encoding='utf-8')
    print(temp_df.head())


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    #main()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    train_filepath = root_dir+"/data/raw/train.csv"
    basic_filepath = root_dir+"/data/interim/basic_preprocessed.csv"
    advanced_filepath = root_dir+"/data/interim/advanced_preprocessed.csv"
    print("Train file:"+train_filepath)
    do_basic_preprocess(train_filepath, basic_filepath)
