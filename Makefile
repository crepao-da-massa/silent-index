CXX ?= g++
CXXFLAGS ?= -Ofast -DNDEBUG -std=c++20 -march=haswell -mtune=haswell -mavx2 -mfma -flto -Wall -Wextra -Wno-unused-parameter
NOEXCEPT_FLAGS ?= -fno-exceptions -fno-rtti

.PHONY: build test index run clean docker-up docker-down

build:
	mkdir -p build
	$(CXX) $(CXXFLAGS) -fopenmp src/build_index.cpp -lz -o build/build-index
	$(CXX) $(CXXFLAGS) $(NOEXCEPT_FLAGS) -pthread src/server.cpp -o build/server
	$(CXX) $(CXXFLAGS) $(NOEXCEPT_FLAGS) -pthread src/profile.cpp -o build/profile
	$(CXX) $(CXXFLAGS) $(NOEXCEPT_FLAGS) -pthread src/offline_api_profile.cpp -o build/offline-api-profile
	gcc -O3 -DNDEBUG -std=c11 -march=haswell -mtune=haswell -flto src/fd_lb.c -o build/fd-lb

test: build
	./build/server --help >/dev/null 2>&1 || true

index: build
	mkdir -p build/index
	./build/build-index bench/references.json.gz build/index/index.bin 1968 65536 6

profile: build
	./build/profile build/index.bin bench/test-data.json 4 8 12 16 24 32

run: build
	mkdir -p /tmp/silent-sockets
	INDEX_PATH=build/index/index.bin LISTEN=/tmp/silent-sockets/api.sock ./build/server

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down -v

clean:
	rm -rf build
