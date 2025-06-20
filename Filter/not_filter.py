

class not_filter:
    def __init__(self, round, dname, input_shape):
        ...
    def run_filter(self,result_clients, nc_t, part_math_wait, server_round, loss = 0):
        result_cl = {}
        for cid, _ in result_clients.items():
            result_cl[cid] = 1.0
        return result_cl, []