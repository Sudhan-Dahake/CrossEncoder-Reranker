from sentence_transformers import CrossEncoder

# Cross-Encoder re-ranker model
re_ranker = CrossEncoder(model_name="cross-encoder/stsb-roberta-base")


def EmphasizeMiddleTokens(chunk: str, emphasis_factor: int = 1):
    # This is a function implemented considering "Lost in the middle" paper.
    # Basically LLMs tend to emphasize more on the start and end tokens rather than middle tokens,
    # thereby leaving important context aside.

    # splits based on whitespace
    tokens = chunk.split()

    length = len(tokens)

    # No middle token, only start and end token is present in the chunk
    if (length < 3):
        return chunk

    # Now our tokens are divided into 3 sections:
    # Start, middle and end
    middle_token_start = length // 3
    middle_token_end = 2 * (middle_token_start)

    # Here we are putting more emphasis on middle tokens by repeating them.
    final_chunk_tokens = tokens[:middle_token_start] + (
        tokens[middle_token_start:middle_token_end] * emphasis_factor) + tokens[middle_token_end:]

    return ''.join(final_chunk_tokens)


def CrossEncoderRanker(query: str, chunks: list[str]):
    # Emphasizing middle tokens
    emphasized_chunks = [EmphasizeMiddleTokens(chunk) for chunk in chunks]

    # Pairing query and chunks to be used as input for cross-encoder
    inputs = [(query, chunk) for chunk in emphasized_chunks]

    # Passing the input to Cross-encoder to get scores for each pair (query, chunk)
    scores = re_ranker.predict(inputs)

    # Sorting the chunks in descending order.
    # That is the most relevant is the first element
    sorted_chunks = [chunk for _, chunk in sorted(
        zip(scores, chunks), reverse=True)]

    # Now rearranging them again based on the strategy discussed in "Lost in the middle" paper
    # That is the most relevant will be the first element, the second most relevant will be the last element
    # The 3rd most relevant will be the 2nd element and so on.
    # This is done because LLMs tend to focus more on the first and last tokens
    ranked_chunks = []
    front_index = 0
    back_index = -1

    for i, chunk in enumerate(sorted_chunks):
        if (i % 2 == 0):
            ranked_chunks.insert(front_index, chunk)
            front_index += 1

        else:
            ranked_chunks.insert(back_index, chunk)
            back_index += -1

    return ranked_chunks[::-1]


def RunTests():
    # A list of dictionaries
    test_cases: dict[str, list[str]] = [
        {
            "query": "What are the benefits of renewable energy?",
            "chunks": [
                "Renewable energy sources include solar, wind, and hydro power.",
                "They help reduce greenhouse gas emissions.",
                "Renewable energy can be more cost-effective in the long run.",
                "Chocolates are sweet.",
                "It reduces dependency on fossil fuels.",
                "Renewable energy systems can be installed in remote areas.",
                "They have a smaller environmental footprint."
            ]
        },
        {
            "query": "How does blockchain technology work?",
            "chunks": [
                "Blockchain is a decentralized ledger technology.",
                "Transactions are recorded in blocks.",
                "Each block is linked to the previous one using cryptography.",
                "Mitochondria is the powerhouse of the cell.",
                "Blockchain is used in various applications like cryptocurrencies.",
                "Mining is the process of adding new blocks to the blockchain.",
                "Blockchain technology faces challenges like scalability."
            ]
        },
        {
            "query": "What are the benefits of a balanced diet?",
            "chunks": [
                "A balanced diet includes a variety of nutrients.",
                "It can help maintain a healthy weight.",
                "Eating a balanced diet reduces the risk of chronic diseases.",
                "Heat",
                "A balanced diet supports proper bodily functions.",
                "It can be challenging to maintain consistently.",
                "Proper nutrition is essential for growth and development."
            ]
        }
    ]

    for i, test_case in enumerate(test_cases):
        query = test_case["query"]
        chunks = test_case["chunks"]
        ranked_chunks = CrossEncoderRanker(query, chunks)

        print(f"Test {i + 1}: {query}")
        for chunk in ranked_chunks:
            print(f"- {chunk}")
        print("\n" + "="*50 + "\n")





if __name__ == '__main__':
    # Test
    RunTests()