import requests
import multiprocessing
from cloudmesh.common.Benchmark import Benchmark
from cloudmesh.common.util import HEADING
from multiprocessing import Pool
import sys
import io

def test_download_data(args):
    client_file = args[0]
    server_file = args[1]
    id = args[2]
    HEADING()
    Benchmark.Start()
    params = {'id': str(id)}
    r = requests.get(f"https://us-east1-anthony-orlowski.cloudfunctions.net/eigenfaces_download_data_http", params=params)
    Benchmark.Stop()
    assert r.status_code == 200
    with io.open(server_file, 'a') as f:
        f.write(r.text)
    f.close()
    print_benchmark(client_file)

def test_train(args):
    client_file = args[0]
    server_file = args[1]
    id = args[2]
    HEADING()
    Benchmark.Start()
    params = {'id': str(id)}
    r = requests.get(f"https://us-east1-anthony-orlowski.cloudfunctions.net/eigenfaces_train_http", params=params)
    Benchmark.Stop()
    assert r.status_code == 200
    with io.open(server_file, 'a') as f:
        f.write(r.text)
    f.close()
    print_benchmark(client_file)

def test_upload(args):
    client_file = args[0]
    server_file = args[1]
    id = args[2]
    HEADING()
    Benchmark.Start()
    params = {'id': str(id)}
    files = {'example_image.jpg': open('example_image.jpg', 'rb')}
    r = requests.post(url=f"https://us-east1-anthony-orlowski.cloudfunctions.net/eigenfaces_upload_http", files=files, params=params)
    Benchmark.Stop()
    assert r.status_code == 200
    with io.open(server_file, 'a') as f:
        f.write(r.text)
    f.close()
    print_benchmark(client_file)

def test_predict(args):
    client_file = args[0]
    server_file = args[1]
    id = args[2]
    HEADING()
    Benchmark.Start()
    params = {'id': str(id)}
    r = requests.get(f"https://us-east1-anthony-orlowski.cloudfunctions.net/eigenfaces_predict_http", params=params)
    Benchmark.Stop()
    assert r.status_code == 200
    with io.open(server_file, 'a') as f:
        f.write(r.text)
    f.close()
    print_benchmark(client_file)

def print_benchmark(client_file):
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    Benchmark.print()
    result = new_stdout.getvalue()
    sys.stdout = old_stdout
    with io.open(client_file, 'a') as f:
        f.write(result)
    f.close()

def run_all_tests_parallel():
    functions = [test_train, test_upload, test_predict]
    for f in functions:
        num = 30
        with Pool(num) as p:
            client_file = f'cold-start-client-{sys.argv[1]}'
            server_file = f'cold-start-server-{sys.argv[1]}'
            args = [(client_file, server_file, i) for i in range(num)]
            p.map(f, args)
        num = 30
        with Pool(num) as p:
            client_file = f'warm-start-client-{sys.argv[1]}'
            server_file = f'warm-start-server-{sys.argv[1]}'
            args = [(client_file, server_file, i) for i in range(num)]
            p.map(f, args)

if __name__ == "__main__":
    test_train(("client.txt", "server.txt","1"))
    #run_all_tests_parallel()
    #for i in range(30):
    #    client_file = f'cold-start-client-{sys.argv[1]}.txt'
    #    server_file = f'cold-start-server-{sys.argv[1]}.txt'
    #    args = []
    #    test_download_data()