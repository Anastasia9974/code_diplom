import math

class RDistributionForParam:
    def __init__(self, common_active_neurons_clients, num_client, num_server):
        self.common_active_neurons_clients = common_active_neurons_clients
        self.num_client = num_client
        self.r_criterion_cl = {}
        self.result_client = {}
        self.avg = 0
        self.std = 0
        self.f = self.num_client - 2
        #self.table_val = [1.869, 1.955] # Для 5 клиентов значимость 0.05, 0.01

        # if num_server == 1:
        #     self.table_val = [1.406, 1.414]
        # else:
        #     self.table_val = [1.412, 1.414]

        self.table_val = [1.397, 1.414]


    def definition_average(self):
        self.avg = sum(val for key, val in self.common_active_neurons_clients.items()) / self.num_client

    def definition_deviation(self):
        total_otcl = 0
        for _, i in self.common_active_neurons_clients.items():
            total_otcl += pow(i-self.avg, 2)
        self.std = math.sqrt(total_otcl/(self.num_client-1))

    def r_criterion(self):
        for cl, x in self.common_active_neurons_clients.items():
            r = abs( x-self.avg )/( self.std*math.sqrt( (self.num_client-1) / self.num_client ) )
            self.r_criterion_cl[cl] = r
        print(f"r_criterion_cl: {self.r_criterion_cl}")

    def detection_bad_client(self):
        for cl, r in self.r_criterion_cl.items():
            result_client = True
            if r > self.table_val[1]:
                result_client = False
            elif r > self.table_val[0]:
                if abs(r - self.table_val[0]) > abs(r - self.table_val[1]):
                    result_client = False
            self.result_client[cl] = result_client

    def get_result_client(self):
        self.definition_average()
        self.definition_deviation()
        self.r_criterion()
        self.detection_bad_client()
        return self.result_client