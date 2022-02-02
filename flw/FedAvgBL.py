


from http import client
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy import FedAvg

import zmq

import numpy as np

from functools import reduce
import sys, json
import zlib






class FedAvgBL(FedAvg):
    """Configurable FedAvg strategy implementation."""


    

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 1,
        fraction_eval: float = 1,
        min_fit_clients: int = 1,
        min_eval_clients: int = 1,
        min_available_clients: int = 1,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        queueAddress = "tcp://localhost:5555"
    ) -> None:
        """Federated Averaging strategy for FlowChain.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_eval : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_eval_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        eval_fn : Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        """
        super().__init__(fraction_fit, fraction_eval, min_fit_clients, min_eval_clients, min_available_clients, eval_fn, on_fit_config_fn, on_evaluate_config_fn, accept_failures, initial_parameters)
        
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(queueAddress)
        self.generalModelVersion  = 0
        
        
        
        '''self.loop = asyncio.get_event_loop()
        self.client = Client(net_profile="/home/niclr/Scrivania/flw/crypto/connection-profile.json")
        self.org1_admin = self.client.get_user(org_name='org1.example.com', name='Admin')
        self.client.new_channel('mychannel')
        self.peer = self.client.get_peer('peer0.org1.example.com')
        self.queue = Queue(1)
        self.thread = Thread(target=self.EventListener, args=(self.queue,))
        self.thread.start()'''

    '''def EventListener(self, queue):

        def newEvent(res):
            queue.put(res)

        channel_event_hub = ChannelEventHub(self.peer, 'mychannel', self.org1_admin)
        channel_event_hub.registerChaincodeEvent(start=0, ccid='fl', pattern='general_model_published', as_array= True, onEvent=newEvent )
        stream = channel_event_hub.connect()
        self.loop.run_until_complete(stream)'''

    def aggregate(self, weights_results):
        args = {'version': self.generalModelVersion, 'model' : [r.tolist() for r in weights_results[0][0]]}

        #args = {'version' : 0, 'model' : [[[[[1,2,3,4,5]]],[3,4,5,],[[[4,5,6,7]]],[4,5]]]}
        #args = json.dumps(args)

        print(sys.getsizeof(json.dumps(args)))
        print(sys.getsizeof(bytes(json.dumps(args), 'utf-8')))
        print(sys.getsizeof(args))
        self.socket.send_json(args)

        '''response = self.loop.run_until_complete(self.client.chaincode_invoke(
                    requestor=self.org1_admin,
                    channel_name='mychannel',
                    peers=['peer0.org1.example.com', 'peer0.org2.example.com'],
                    fcn = 'PubPart',
                    args=[args],
                    cc_name='fl',
                    cc_type = 'NODE',
                    transient_map=None, # optional, for private data
                    wait_for_event=True, # for being sure chaincode invocation has been commited in the ledger, default is on tx event
                    raise_on_error=True
                    ))'''
        res = self.socket.recv_json()
        
        self.generalModelVersion = res.version
        new_weights : Weights = [np.ndarray(res.model)]
        '''
        num_examples_total = sum([num_examples for _, num_examples in weights_results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in weights_results
        ]

        # Compute average weights of each layer
        weights_prime: Weights = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        '''
        return new_weights
    
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        return weights_to_parameters(self.aggregate(weights_results)), {}