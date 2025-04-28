

class not_filter:
    def __init__(self, round, dname, input_shape):
        ...
    def run_filter(self,result_clients, nc_t, part_math_wait, server_round, loss = 0):
        return [1.0]*len(result_clients), []