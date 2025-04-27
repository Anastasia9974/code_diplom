
class r_distribution_for_param:
    def __init__(self, common_active_neurons_clients, num_client):
        self.common_active_neurons_clients = common_active_neurons_clients
        self.num_client = num_client
        self.r_criterion_cl = []
        self.avg = 0
        self.std = 0

    def parameter_definition(self):
        self.definition_average()
        self.definition_deviation()
        self.r_criterion()

    def definition_average(self):
        ...

    def definition_deviation(self):
        ...

    def r_criterion(self):
        ...