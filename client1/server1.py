import flwr as fl

from FedAvgBL import FedAvgBL

# Start Flower server for three rounds of federated learning
strategy = FedAvgBL(queueAddress="tcp://localhost:5555")
fl.server.start_server(server_address="[::]:8081", config={"num_rounds": 10}, strategy=strategy)