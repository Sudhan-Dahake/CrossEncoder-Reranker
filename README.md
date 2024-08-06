# CrossEncoder Re-Ranker Model

This project implements a Cross-Encoder re-ranker model using the `sentence-transformers` library. The primary objective is to enhance the relevance of text chunks by emphasizing the middle tokens, based on the principles discussed in the paper "Lost in the Middle." The model then re-ranks these text chunks in order of relevance to a given query.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Functions](#functions)
  - [EmphasizeMiddleTokens](#emphasizemiddletokens)
  - [CrossEncoderRanker](#crossencoderranker)
  - [RunTests](#runtests)
- [Testing](#testing)

## Introduction

The project leverages the Cross-Encoder model from the `sentence-transformers` library to rank text chunks based on their relevance to a query. It emphasizes the middle tokens in each chunk to improve the model's attention to important context, addressing the tendency of language models to focus more on the start and end tokens.

## Features

- **Cross-Encoder Model:** Utilizes the `cross-encoder/stsb-roberta-base` model for re-ranking.
- **Token Emphasis:** Implements the "Lost in the Middle" strategy to emphasize middle tokens.
- **Relevance Ranking:** Ranks text chunks based on their relevance to a query.
- **Reordering Strategy:** Rearranges chunks to further enhance relevance based on "Lost in the Middle".

## Installation

To get started with this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/Sudhan-Dahake/CrossEncoder-Reranker.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Functions

### EmphasizeMiddleTokens

```python
def EmphasizeMiddleTokens(chunk: str, emphasis_factor: int = 1) -> str:
```

This function emphasizes the middle tokens in a given text chunk by repeating them based on the `emphasis_factor`. It addresses the tendency of language models to focus more on the start and end tokens, potentially overlooking important context in the middle.

### CrossEncoderRanker

```python
def CrossEncoderRanker(query: str, chunks: list[str]) -> list[str]:
```

This function ranks text chunks based on their relevance to a given query. It first emphasizes the middle tokens in each chunk, then uses the Cross-Encoder model to score and rank the chunks. The ranked chunks are reordered based on a strategy to further enhance relevance.

### RunTests

```python
def RunTests():
```

This function runs a series of test cases to demonstrate the effectiveness of the Cross-Encoder re-ranker model. It prints the ranked chunks for each test case.

## Testing

To run the tests, execute the script:

```bash
python Cross_Encoder.py
```
