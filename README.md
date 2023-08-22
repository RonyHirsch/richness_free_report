# More than Words

This repository contains all the parsing and processing scripts used in the manuscript: "More than words: Can free reports adequately measure the richness of perception?". 
This work describes a series of three experiments run in two batches (exploratory, replication), where participants were exposed to briefly-presented images, and then asked to provide five words describing their experience. 


All experiments, parsing and processing codes were programmed by [Rony Hirschhorn](https://github.com/RonyHirsch).

## Processing Codes

Data processing is done separately per experiment. 

### Main Experiments

#### Parsing
The processing module for the main experiments is managed by [manage_analysis.py](manage_analysis.py):
* The first stage is to pre-process all the raw data files into something coherent, and attribute each participant with all the responses they made during the experiment. This is done by calling the [process_raw_data.py](process_raw_data.py) module's manager function (manage_preprocessing).
* Then, once all data is loaded from the files, the actual parsing, processing and descriptive stats can begin. This is done by the [analyze_data.py](analyze_data.py) module (by calling its manager function, manage_data_analysis).

*NOTE*: during word parsing, we use nltk's stop words (from nltk.corpus import stopwords). This requires [installing nltk](https://www.nltk.org/install.html) and downloading stop words:
* Either by following the import line with ```nltk.download('stopwords')```
* Or straight from the commandline: ```python -m nltk.downloader stopwords```

#### Aggregation
Once done processing each experiment separately, the results are then aggregated for plotting and more descriptives purposes. This is a separate process, that is managed by the [aggregate_results.py](aggregate_results.py) module. 

#### Additional Datasets
Furthermore, our experiments follow the design of [this experiment](10.12688/f1000research.75364.2). In case one is interested in stats about their data calculated in our scripts, the [analyze_Chuyin_data.py](analyze_Chuyin_data.py) module reshapes the data from this experiment to match the shape of our experimental outputs; then, it runs the manage_analysis module.

### Control Experiments
In addition, two separate control experiments were performed in our study. To parse and aggregate their results, a module called [control_analysis.py](control_analysis.py) handles the processing of both experiments, separately. 
