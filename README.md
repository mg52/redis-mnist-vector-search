# Redis Vector Search with MNIST Dataset in Go

This repository demonstrates how to use Redis for vector similarity search on the [MNIST dataset](https://yann.lecun.com/exdb/mnist/) ([csv can be found here](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/)), leveraging Redis' RediSearch and RedisJSON modules with a Go implementation. The project achieves 96% accuracy, correctly predicting 9691 out of 10000 test images, with an efficient average search duration of 29 ms, making Redis a powerful solution for high-dimensional data search.
 
## Setup

### Step 1: Run Redis in Docker

First, start Redis with `RediSearch` and `RedisJSON` modules using Docker. The command below also sets up persistent data storage and a password.

```bash
mkdir $HOME/redisdata
docker run -d --name redis-stack-server -p 6379:6379 -v $HOME/redisdata:/data -e REDIS_ARGS="--requirepass thepassword --appendonly yes --appendfsync always" redis/redis-stack-server:latest
```

### Step 2: Clone the Repository
Clone this GitHub repository to your local machine:
```bash
git clone https://github.com/mg52/redis-mnist-vector-search.git
cd redis-mnist-vector-search
```

### Step 3: Clone the Repository
Install the required Go packages:
```bash
go mod tidy
```

### Step 4: Download MNIST CSV
Download MNIST CSV file and replace the file paths (os.Open paths) in the code.

### Step 5: Run the Code
Run the Go application:
```bash
go run .
```

## Code Explanation

### 1. Creating Index
We are indexing vectors with 784 dimensions for MNIST data pixels in float32 using a FLAT vector index with 6 initial vectors with L2(Euclidean) distance metric.
```bash
createIndex := []interface{}{
  "FT.CREATE", "mnist_index", "ON", "JSON",
  "PREFIX", "1", "number:",
  "SCHEMA", "$.embedding", "AS", "embedding",
  "VECTOR", "FLAT", "6", "DIM", "784",
  "DISTANCE_METRIC", "L2", "TYPE", "FLOAT32",
}
```

### 2. Storing MNIST data as JSON in Redis
MNIST dataset consists 70000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. We are converting those numbers in `mnist_train.csv` file to float32, dividing the pixel value with 255 and then storing as a JSON object in embedding field.
```bash
// Create JSON data for Redis
jsonData := fmt.Sprintf(`{"result": %d, "embedding": [%s]}`, result, embedding)

// Execute the JSON.SET command directly in Redis
key := fmt.Sprintf("number:%d:%d", i, result)
err = rdb.Do(ctx, "JSON.SET", key, "$", jsonData).Err()
if err != nil {
  return err
}
```


### 3. Performing a Vector Search
We are performing a K-Nearest Neighbors (KNN) search on the mnist_index, finding the closest 1 vector to the provided embedding (embeddingBytes), sorts the results by distance (dist), and uses RediSearch's query dialect 2 for parameter handling. This embeddingBytes comes from `mnist_test.csv` file.
```bash
searchQuery := []interface{}{
  "FT.SEARCH",                           
  "mnist_index",                        
  "*=>[KNN 1 @embedding $blob AS dist]",
  "SORTBY", "dist",                      
  "PARAMS", "2", "blob", embeddingBytes, 
  "DIALECT", "2", 
}
```

## Output

When you run the code it prints below output:
```bash
...
...
Stored JSON for number:59996:3
Stored JSON for number:59997:5
Stored JSON for number:59998:6
Stored JSON for number:59999:8
All data has been stored in Redis.
Test image 0: expected = 7, found = 7 in 81ms
Test image 1: expected = 2, found = 2 in 31ms
Test image 2: expected = 1, found = 1 in 30ms
Test image 3: expected = 0, found = 0 in 30ms
...
...
Test image 9997: expected = 4, found = 4 in 30ms
Test image 9998: expected = 5, found = 5 in 30ms
Test image 9999: expected = 6, found = 6 in 29ms
Number of Correct guess = 9691
Number of Wrong guess = 309
Accuracy = 96%
Redis Vector Search Min Duration = 29ms
Redis Vector Search Max Duration = 99ms
Redis Vector Search Average Duration = 29ms
```



