# 0. Introduction about parsed dataset

To implement the project, we parsed sample of datasets from kaggle which include metadata (all information about dataset) and files of dataset in zip archive or only metadata.

[Storage with datasets from kaggle](https://disk.yandex.ru/d/So2CxXs5J2Ol5g)

# 1. First parse script

## 1.1. Structure of files

- 📦 datasets
  - 📂 kaggle
      - 📂 {kaggle user name}
          - 📂 {kaggle internal dataset name}
              - 📚 {some dataset files}.zip
              - 📄 {information about dataset from kaggle}.json
          - 📂 {kaggle internal dataset name}
              - 📚 {some dataset files}.zip
              - 📄 {information about dataset from kaggle}.json
        - 📂 {kaggle internal dataset name}
            - 📚 {some dataset files}.zip
            - 📄 {information about dataset from kaggle}.json

## 1.2. Parse settings

list of queries which used to parse these datasets:
* 'regression'
* 'text classification'
* 'recommendation'
* 'Time Series Analysis'
* 'Natural Language Processing'
* 'Data analysis and visualization'

Search for datasets in the API as in the web interface of kaggle.
So I took 20 pages with 20 datasets for all queries sorted by votes with minimum=1MB and maximum=100MB.

At this stage, 659 datasets with metadata were eventually downloaded.

# 2. Second parse script

A set of queries for parsing was selected. For which only metadata was downloaded, without datasets. The set of queries can be found here: `notebooks/queries`