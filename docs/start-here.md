# Anserini: Start Here

This page provides the entry point for an introduction to information retrieval (i.e., search).
It also serves as an [onboarding path](https://github.com/lintool/guide/blob/master/ura.md) for University of Waterloo undergraduate (and graduate) students who wish to join my research group.

As a high-level tip for anyone going through these exercises: try to understand what you're actually doing, instead of simply [cargo culting](https://en.wikipedia.org/wiki/Cargo_cult_programming) (i.e., blindly copying and pasting commands into a shell).
By this, I mean, actually _read_ the surrounding explanations, understand the purpose of the commands, and use this guide as a springboard for additional explorations (for example, dig deeper into the code).

**Learning outcomes** for this guide:

+ Understand the definition of the retrieval problem in terms of the core concepts of queries, collections, and relevance.
+ Understand at a high level how retrieval systems are evaluated with queries and relevance judgments.
+ Download the MS MARCO passage ranking test collection and perform basic manipulations on the downloaded files.
+ Connect the contents of specific files in the download package to the concepts referenced above.

## The Retrieval Problem

Let's start at the top:
What's the problem we're trying to solve?

This is the definition I typically give:

> Given an information need expressed as a query _q_, the text retrieval task is to return a ranked list of _k_ texts {_d<sub>1</sub>_, _d<sub>2</sub>_ ... _d<sub>k</sub>_} from an arbitrarily large but finite collection
of texts _C_ = {_d<sub>i</sub>_} that maximizes a metric of interest, for example, nDCG, AP, etc.

This problem has been given various names, e.g., the search problem, the information retrieval problem, the text ranking problem, the top-_k_ document retrieval problem, etc.
In most contexts, "ranking" and "retrieval" are used interchangeably.
Basically, this is what _search_ (i.e., information retrieval) is all about.

Let's try to unpack the definition a bit.

A **"query"** is a representation of an information need (i.e., the reason you're looking for information in the first place) that serves as the input to a retrieval system.

The **"collection"** is what you're retrieving from (i.e., searching).
Some people say "corpus" (plural, "corpora", not "corpuses"), but the terms are used interchangeably.
A "collection" or "corpus" comprises "documents".
In standard parlance, a "document" is used generically to refer to any discrete information object that can be retrieved.
We call them "documents" even though in reality they may be webpages, passages, PDFs, Powerpoint slides, Excel spreadsheets, or even images, audio, or video.

The output of retrieval is a ranking of documents (i.e., a ranked list, or just a sorted list).
Documents are identified by unique ids, and so a ranking is simply a list of ids.
The document contents can serve as input to downstream processing, e.g., fed into the prompt of a large language model as part of retrieval augmentation or generative question answering.

**Relevance** is perhaps the most important concept in information retrieval.
The literature on relevance goes back at least fifty years and the notion is (surprisingly) difficult to pin down precisely.
However, at an intuitive level, relevance is a relationship between an information need and a document, i.e., is this document relevant to this information need?
Something like, "does this document contain the information I'm looking for?"

Sweeping away a lot of complexity... relevance can be binary (i.e., not relevant, relevant) or "graded" (think Likert scale, e.g., not relevant, relevant, highly relevant).

## The Evaluation Problem

How do you know if a retrieval system is returning good results?
How do you know if _this_ retrieval system is better than _that_ retrieval system?

Well, imagine if you had a list of "real-world" queries, and someone told you which documents were relevant to which queries.
Amazingly, these artifacts exist, and they're called **relevance judgments** or **qrels**.
Conceptually, they're triples along these lines:

```
q1 doc23 0
q1 doc452 1
q1 doc536 0
q2 doc97 0
...
```

That is, for `q1`, `doc23` is not relevant, `doc452` is relevant, and `doc536` is not relevant; for `q2`, `doc97` is not relevant...

Now, given a set of queries, you feed each query into your retrieval system and get back a ranked list of document ids.

The final thing you need is a **metric** that quantifies the "goodness" of the ranked list.
One easy-to-understand metric is precision at 10, often written P@10.
It simply means: of the top 10 documents, what fraction are relevant according to your qrels?
For a query, if five of them are relevant, you get a score of 0.5; if nine of them are relevant, you get a score of 0.9.
You compute P@10 per query, and then average across all queries.

Information retrieval researchers have dozens of metrics, but a detailed explanation of each isn't important right now... 
just recognize that _all_ metrics are imperfect, but they try to capture different aspects of the quality of a ranked list in terms of containing relevant documents.
For nearly all metrics, though, higher is better.

So now with a metric, we have the ability to measure (i.e., quantify) the quality of a system's output.
And with that, we have the ability to compare the quality of two different systems or two model variants.

With a metric, we have the ability to iterate incrementally and build better retrieval systems (e.g., with machine learning).
Oversimplifying (of course), information retrieval is all about making the metric go up.

Oh, where do these magical relevance judgments (qrels) come from?
Well, that's the story for another day...

<details>
<summary>Additional readings...</summary>

This is a very high-level summary of core concepts in information retrieval.
More nuanced explanations are presented in our book [Pretrained Transformers for Text Ranking: BERT and Beyond](
https://link.springer.com/book/10.1007/978-3-031-02181-7).
If you can access the book (e.g., via your university), then please do, since it helps get our page views up.
However, if you're paywalled, a pre-publication version is available [on arXiv](https://arxiv.org/abs/2010.06467) for free.

The parts you'll want to read are Section 1.1 "Text Ranking Problems" and all of Chapter 2 "Setting the Stage".

When should you do these readings?
That's a good question:
If you absolutely want to know more _right now_, then go for it.
Otherwise, I think it's probably okay to continue along the onboarding path... although you'll need to circle back and get a deeper understanding of these concepts if you want to get into information retrieval research "more seriously".
</details>

## A Tour of MS MARCO

Bringing together everything we've discussed so far, a test collection consists of three elements:

+ a collection (or corpus) of documents
+ a set of queries
+ relevance judgments (or qrels), which tell us which documents are relevant to which queries

Here, we're going to introduce the [MS MARCO passage ranking test collection](https://microsoft.github.io/msmarco/).

If you haven't cloned the [anserini](https://github.com/castorini/anserini) repository already, clone it and get its `tools` submodule:
```bash
git clone https://github.com/castorini/anserini.git
cd anserini
git submodule update --init --recursive
```

In these instructions we're going to use Anserini's root directory as the working directory.
First, we need to download and extract the data:

```bash
mkdir collections/msmarco-passage

wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz -P collections/msmarco-passage

# Alternative mirror:
# wget https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/data/collectionandqueries.tar.gz -P collections/msmarco-passage

tar xvfz collections/msmarco-passage/collectionandqueries.tar.gz -C collections/msmarco-passage
```

To confirm, `collectionandqueries.tar.gz` should have MD5 checksum of `31644046b18952c1386cd4564ba2ae69`:
```bash
md5sum collections/msmarco-passage/collectionandqueries.tar.gz
```

If you peak inside the collection:

```bash
head collections/msmarco-passage/collection.tsv
```

You'll see that `collection.tsv` contains the documents that we're searching.
Note that generically we call them "documents" but in truth they are passages; we'll use the terms interchangeably.

Each line represents a passage:
the first column contains a unique identifier for the passage (called the `docid`) and the second column contains the text of the passage itself.

Next, we need to do a bit of data munging to get the collection into something Anserini can easily work with, which is a jsonl format (where we have one json object per line):

```bash
python tools/scripts/msmarco/convert_collection_to_jsonl.py \
  --collection-path collections/msmarco-passage/collection.tsv \
  --output-folder collections/msmarco-passage/collection_jsonl
```

The above script should generate 9 jsonl files in `collections/msmarco-passage/collection_jsonl`, each with 1M lines (except for the last one, which should have 841,823 lines).
Look inside a file to see the json format we use.
The entire collection is now something like this:

```bash
$ wc collections/msmarco-passage/collection_jsonl/* 
 1000000 58716381 374524070 collections/msmarco-passage/collection_jsonl/docs00.json
 1000000 59072018 377845773 collections/msmarco-passage/collection_jsonl/docs01.json
 1000000 58895092 375856044 collections/msmarco-passage/collection_jsonl/docs02.json
 1000000 59277129 377452947 collections/msmarco-passage/collection_jsonl/docs03.json
 1000000 59408028 378277584 collections/msmarco-passage/collection_jsonl/docs04.json
 1000000 60659246 383758389 collections/msmarco-passage/collection_jsonl/docs05.json
 1000000 63196730 400184520 collections/msmarco-passage/collection_jsonl/docs06.json
 1000000 56920456 364726419 collections/msmarco-passage/collection_jsonl/docs07.json
  841823 47767342 306155721 collections/msmarco-passage/collection_jsonl/docs08.json
 8841823 523912422 3338781467 total
```

As an aside, data munging along these lines is a very common data preparation operation.
Collections rarely come in _exactly_ the format that your tools except, so you'll be frequently writing lots of small scripts that munge data to convert from one format to another.

Similarly, we'll also have to do a bit of data munging of the queries and the qrels.
We're going to retain only the queries that are in the qrels file: 

```bash
python tools/scripts/msmarco/filter_queries.py \
  --qrels collections/msmarco-passage/qrels.dev.small.tsv \
  --queries collections/msmarco-passage/queries.dev.tsv \
  --output collections/msmarco-passage/queries.dev.small.tsv
```

The output queries file `collections/msmarco-passage/queries.dev.small.tsv` should contain 6980 lines.
Verify with `wc`.

Check out its contents:

```bash
$ head collections/msmarco-passage/queries.dev.small.tsv
1048585	what is paula deen's brother
2	 Androgen receptor define
524332	treating tension headaches without medication
1048642	what is paranoid sc
524447	treatment of varicose veins in legs
786674	what is prime rate in canada
1048876	who plays young dr mallard on ncis
1048917	what is operating system misconfiguration
786786	what is priority pass
524699	tricare service number
```

These are the queries in the development set of the MS MARCO passage ranking test collection.
The first field is a unique identifier for the query (called the `qid`) and the second column is the query itself.
These queries are taken from Bing search logs, so they're "realistic" web queries in that they may be ambiguous, contain typos, etc.

Okay, let's now cross reference the `qid` with the relvance judgments, i.e., the qrels file:

```bash
$ grep 1048585 collections/msmarco-passage/qrels.dev.small.tsv 
1048585	0	7187158	1
```

The above is the standard format for a qrels file.
The first column is the `qid`;
the second column is (almost) always 0 (it's a historical artifact dating back decades);
the third column is a `docid`;
the fourth colum provides the relevance judgment itself.
In this case, 0 means "not relevant" and 1 means "relevant".
So, this entry says that the document with id 7187158 is relevant to the query with id 1048585.

Well, how do we get the actual contents of document 1048585?
The simplest way is to grep through the collection itself:

```bash
$ grep 7187158 collections/msmarco-passage/collection.tsv
7187158	Paula Deen and her brother Earl W. Bubba Hiers are being sued by a former general manager at Uncle Bubba'sâ¦ Paula Deen and her brother Earl W. Bubba Hiers are being sued by a former general manager at Uncle Bubba'sâ¦
```

We see here that, indeed, the passage above is relevant to the query (i.e., provides information that answers the question).
Note that this particular passage is a bit dirty (garbage characters, dups, etc.)... but that's pretty much a fact of life when you're dealing with the web.

Before proceeding, try the same thing with a few more queries: map the queries to the relevance judgments to the actual documents.

How big is the MS MARCO passage ranking test collection, btw?
Well, we've just seen that there are 6980 training queries.
For those, we have 7437 relevance judgments:

```bash
$ wc collections/msmarco-passage/qrels.dev.small.tsv  
7437   29748  143300 collections/msmarco-passage/qrels.dev.small.tsv
````

This means that we have only about one relevance judgments per query.
We call these **sparse judgments**, i.e., where we have relatively few relevance judgments per query (here, just about one relevance judgment per query).
In other cases, where we have many relevance judgments per query (potentially hundreds or even more), we call those **dense judgments**.
There are important implications when using sparse vs. dense judgments, but that's for another time...

This is just looking at the development set.
Now let's look at the training set:

```bash
$ wc collections/msmarco-passage/qrels.train.tsv               
532761 2131044 10589532 collections/msmarco-passage/qrels.train.tsv
```

Wow, there are over 532k relevance judgments in the dataset!
(Yes, that's big!)
It's sufficient... for example... to _train_ neural networks (transformers) to perform retrieval!
And, indeed, MS MARCO is perhaps the most common starting point for work in building neural retrieval models.
But that's for some other time....

Okay, go back and look at the learning outcomes at the top of this page.
By now you should be able to connect the concepts we introduced to how they manifest in the MS MARCO passage ranking test collection.

From here, you're now ready to proceed to try and reproduce the [BM25 Baselines for MS MARCO Passage Ranking
](experiments-msmarco-passage.md).

Before you move on, however, add an entry in the "Reproduction Log" at the bottom of this page, following the same format: use `yyyy-mm-dd`, make sure you're using a commit id that's on the main trunk of Anserini, and use its 7-hexadecimal prefix for the link anchor text.

## Reproduction Log[*](reproducibility.md)
