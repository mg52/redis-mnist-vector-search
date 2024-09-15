package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/go-redis/redis/v8"
)

var ctx = context.Background()
var minDuration, maxDuration, totalDuration int64

// CreateIndex creates redis index for
// FT.CREATE mnist_index ON JSON PREFIX 1 number: SCHEMA $.embedding AS embedding VECTOR FLAT 6 DIM 784 DISTANCE_METRIC L2 TYPE FLOAT32
func CreateIndex(rdb *redis.Client) error {
	createIndex := []interface{}{
		"FT.CREATE", "mnist_index", "ON", "JSON",
		"PREFIX", "1", "number:",
		"SCHEMA", "$.embedding", "AS", "embedding",
		"VECTOR", "FLAT", "6", "DIM", "784",
		"DISTANCE_METRIC", "L2", "TYPE", "FLOAT32",
	}

	// Execute the FT.SEARCH command using Do()
	_, err := rdb.Do(ctx, createIndex...).Result()
	return err
}

func StoreData(rdb *redis.Client) error {
	// Open the MNIST CSV file
	file, err := os.Open("mnist_train.csv")
	if err != nil {
		return err
	}
	defer file.Close()

	// Create a CSV reader
	reader := csv.NewReader(bufio.NewReader(file))

	// Read each record from the CSV file
	records, err := reader.ReadAll()
	if err != nil {
		return err
	}

	// Iterate over each row in the CSV file
	for i, record := range records {
		// The first value is the result (the number)
		result, err := strconv.Atoi(record[0])
		if err != nil {
			return err
		}

		// The rest are pixel values
		pixelValues := record[1:]

		// Convert pixel values to float32 and normalize them by dividing by 255
		var pixelStrings []string
		for _, pixel := range pixelValues {
			pixelInt, err := strconv.Atoi(pixel)
			if err != nil {
				return err
			}
			// If the pixel value is 0, directly append "0", else format as float32 with 6 decimals
			if pixelInt == 0 {
				pixelStrings = append(pixelStrings, "0")
			} else {
				pixelFloat := float32(pixelInt) / 255.0
				pixelStrings = append(pixelStrings, fmt.Sprintf("%.6f", pixelFloat))
			}
		}
		embedding := strings.Join(pixelStrings, ",")

		// Create JSON data for Redis
		jsonData := fmt.Sprintf(`{"result": %d, "embedding": [%s]}`, result, embedding)

		// Execute the JSON.SET command directly in Redis
		key := fmt.Sprintf("number:%d:%d", i, result)
		err = rdb.Do(ctx, "JSON.SET", key, "$", jsonData).Err()
		if err != nil {
			return err
		}
		fmt.Printf("Stored JSON for number:%d:%d\n", i, result)
	}

	fmt.Println("All data has been stored in Redis.")
	return nil
}

func SearchData(rdb *redis.Client) error {
	// Open the MNIST test CSV file
	file, err := os.Open("mnist_test.csv")
	if err != nil {
		return err
	}
	defer file.Close()

	// Create a CSV reader
	reader := csv.NewReader(bufio.NewReader(file))

	// Read each record from the CSV file
	records, err := reader.ReadAll()
	if err != nil {
		return err
	}

	correctGuess := 0
	wrongGuess := 0
	// Iterate over each row in the test CSV file
	for i, record := range records {
		// The first value is the expected result (the label)
		expectedResult, err := strconv.Atoi(record[0])
		if err != nil {
			return err
		}

		// The rest are pixel values
		pixelValues := record[1:]

		// Convert pixel values to float32 and normalize them by dividing by 255
		var embedding []float32
		for _, pixel := range pixelValues {
			pixelInt, err := strconv.Atoi(pixel)
			if err != nil {
				return err
			}
			// Normalize the pixel value
			pixelFloat := float32(pixelInt) / 255.0
			embedding = append(embedding, pixelFloat)
		}

		// Perform the FT.SEARCH query using the normalized embedding
		foundLabel, duration, err := searchVectorInRedis(rdb, embedding)
		if err != nil {
			return err
		}
		if duration < minDuration {
			minDuration = duration
		}
		if duration > maxDuration {
			maxDuration = duration
		}
		totalDuration += duration
		// Print the expected result and the found label
		fmt.Printf("Test image %d: expected = %d, found = %d in %dms\n", i, expectedResult, foundLabel, duration)
		if expectedResult == foundLabel {
			correctGuess++
		} else {
			wrongGuess++
		}
	}
	fmt.Printf("Number of Correct guess = %d\n", correctGuess)
	fmt.Printf("Number of Wrong guess = %d\n", wrongGuess)
	fmt.Printf("Accuracy = %d%%\n", 100*correctGuess/(wrongGuess+correctGuess))
	fmt.Printf("Redis Vector Search Min Duration = %dms\n", minDuration)
	fmt.Printf("Redis Vector Search Max Duration = %dms\n", maxDuration)
	fmt.Printf("Redis Vector Search Average Duration = %dms\n", totalDuration/int64(len(records)))

	return nil
}

func convertFloat32ArrayToBlob(vector []float32) ([]byte, error) {
	buf := new(bytes.Buffer)
	for _, v := range vector {
		err := binary.Write(buf, binary.LittleEndian, v)
		if err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

// searchVectorInRedis performs an FT.SEARCH query on the mnist_index using the embedding
func searchVectorInRedis(rdb *redis.Client, embedding []float32) (int, int64, error) {
	// Convert the embedding to a byte slice (binary format)
	embeddingBytes, err := convertFloat32ArrayToBlob(embedding)
	if err != nil {
		return 0, 0, err
	}

	searchQuery := []interface{}{
		"FT.SEARCH",                           // Explicitly using the FT.SEARCH command
		"mnist_index",                         // Index name
		"*=>[KNN 1 @embedding $blob AS dist]", // KNN search query
		"SORTBY", "dist",                      // Sort by distance
		"PARAMS", "2", "blob", embeddingBytes, // Params: search vector blob
		"DIALECT", "2", // RedisSearch dialect 2
	}

	start := time.Now()

	// Execute the FT.SEARCH command using Do()
	result, err := rdb.Do(ctx, searchQuery...).Result()
	duration := time.Since(start).Milliseconds()
	if err != nil {
		return 0, 0, err
	}

	items, ok := result.([]interface{})
	if !ok || len(items) == 0 {
		return 0, 0, fmt.Errorf("unexpected result format")
	}

	parts := strings.Split(items[1].(string), ":")

	// Get the last part (which should be the digit)
	lastPart := parts[len(parts)-1]

	// Convert the last part to an integer
	parsedInt, err := strconv.Atoi(lastPart)
	if err != nil {
		return 0, 0, err
	}

	return parsedInt, duration, nil
}

func main() {
	minDuration = 999999
	maxDuration = 0
	totalDuration = 0
	// Connect to Redis
	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379", // Replace with your Redis server address
		Password: "thepassword",    // Set Redis password if needed
		DB:       0,                // Use default DB
	})

	defer rdb.Close()

	err := CreateIndex(rdb)
	if err != nil {
		if strings.Contains(err.Error(), "Index already exists") {
			slog.Warn("Index already exists.")
		} else {
			slog.Error("Could not create search index.", slog.String("error", err.Error()))
			os.Exit(1)
		}
	} else {
		slog.Info("Index Created.")
	}

	err = StoreData(rdb)
	if err != nil {
		slog.Error("Could not store data.", slog.String("error", err.Error()))
		os.Exit(1)
	}

	err = SearchData(rdb)
	if err != nil {
		slog.Error("Could not search data.", slog.String("error", err.Error()))
		os.Exit(1)
	}
}
