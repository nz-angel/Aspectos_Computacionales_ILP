from itertools import product
from math import sqrt, inf


class ProblemInstance:
    """
    Clase utilizada para almacenar los datos del problema. El mismo viene dado en un archivo .txt con un formato
    especial.
    """

    def __init__(self, filepath):
        self.fixed_cost, self.var_cost, self.cap, node_coords, self.demand = self._read_file(filepath)
        self.dist_matrix = self._calculate_distance_matrix(node_coords)

    def arc_cost(self, k, i, j):
        return self.var_cost[k] * self.dist_matrix[i][j]

    @staticmethod
    def _read_file(filepath):
        vehicle_fixed_cost = {}
        vehicle_var_cost = {}
        vehicle_capacity = {}
        node_coords = {}
        node_demand = {}
        reading_nodes = False
        with open(filepath, 'r') as f:
            for line in f.readlines():
                if line != '\n':
                    if not line[0].isalpha():
                        line = line.strip().split(' ')
                        try:
                            line_data = list(map(int, line))
                        except ValueError:
                            line_data = list(map(int, line[:3])) + [float(line[-1])]
                        k = line_data[0]
                        if not reading_nodes:
                            vehicle_capacity[k] = line_data[1]
                            vehicle_fixed_cost[k] = line_data[2]
                            vehicle_var_cost[k] = line_data[3]
                        else:
                            node_coords[k] = (line_data[1], line_data[2])
                            node_demand[k] = line_data[3]
                else:
                    reading_nodes = True
        return vehicle_fixed_cost, vehicle_var_cost, vehicle_capacity, node_coords, node_demand

    @staticmethod
    def _calculate_distance_matrix(node_coords):
        dist_matrix = {n: {} for n in node_coords}
        for n, m in product(node_coords, repeat=2):
            dist_matrix[n][m] = sqrt((node_coords[n][0] - node_coords[m][0])**2 + (node_coords[n][1] - node_coords[m][1])**2)
        return dist_matrix


class Path:
    """ Clase utilizada para representar un camino, en el sentido de secuencia de nodos, de cierto tipo de vehículo k.
    Comienza siempre con el nodo 0 correspondiente al depósito. No es obligatorio que termine en 0. Su costo se mide
    como el costo variable del tipo de vehículo k multiplicado por la longitud de los arcos que lo componen. """

    def __init__(self, k):
        self.sequence = [0]     # Secuencia de nodos de la ruta
        self.vehicle = k        # Tipo de vehículo
        self.cost = 0           # Costo
        self.demand = 0         # Suma de la demanda de los nodos visitados

    def add_node(self, i, instance):
        self.cost += instance.arc_cost(self.vehicle, self.sequence[-1], i)
        self.sequence.append(i)
        self.demand += instance.demand[i]

    @classmethod
    def from_sequence(cls, sequence, instance, k):
        if sequence[0] != 0:
            raise ValueError('El primer nodo de la secuencia debe ser 0')

        path = cls(k)
        path.sequence = sequence
        for i, j in zip(sequence[:-1], sequence[1:]):
            path.cost += instance.dist_matrix[i][j] * instance.var_cost[k]
            path.demand += instance.demand[j]
        return path


class Route:
    """
    Representa la ruta de un vehículo. Es una clase análoga a Path pero en este caso sí debe comenzar y terminar
    en el nodo 0. Esta pensada para incializarse a partir de una instancia de Path y de los datos del problema.
    """

    def __init__(self):
        self.sequence = []      # Secuencia de nodos visitados
        self.vehicle = None     # Tipo de vehículo que realiza el tour
        self.demand = 0         # Suma de la demanda de los nodos visitados
        self.cost = 0           # Costo de realizar el tour

    @classmethod
    def from_path(cls, path, instance):
        k = path.vehicle
        route = Route()
        route.sequence = path.sequence + [0]
        route.demand = path.demand
        route.vehicle = k
        route.cost = path.cost + instance.arc_cost(k, path.sequence[-1], 0) + instance.fixed_cost[k]
        return route

    @classmethod
    def from_sequence(cls, sequence, instance, k):
        route = cls()
        if sequence[0] != 0 or sequence[-1] != 0:
            raise ValueError('La ruta debe empezar y terminar en el depósito')
        route.sequence = sequence
        route.vehicle = k
        if len(sequence) == 1:
            route.cost = 0
        else:
            route.cost = sum(instance.arc_cost(k, i, j) for i, j in zip(sequence[:-1], sequence[1:])) + \
                         instance.fixed_cost[k]
        route.demand = sum(instance.demand[i] for i in sequence)
        return route

    def reduced_cost(self, pi, lamb):
        return self.cost - sum(pi[j] for j in self.sequence[1:-1]) - lamb[self.vehicle]

    def __eq__(self, other):
        return self.sequence == other.sequence and self.vehicle == other.vehicle


class State:
    """
    Esta clase se utiliza en la resolución de SP(k) con programación dinámica. Cada estado tiene los siguientes
    componentes:
    node: nodo final del camino mínimo desde el depósito
    q: carga acumulada a lo largo del camino
    pred: predecesor de j en el camino
    path: secuencia de nodos del camino
    ub: cota superior de la longitud del camino mínimo desde el depósito hasta j
    lb: cota inferior de la longitud del camino mínimo desde el depósito hasta j
    sub: cota superior del segundo mejor camino hasta j
    """

    def __init__(self, q, j):
        self.node = j
        self.q = q
        self.pred = None
        self.path = []
        self.ub = inf
        self.lb = -inf
        self.sub = inf

    @classmethod
    def warehouse_state(cls):
        state = cls(0, 0)
        state.path = [0]
        state.ub = 0
        state.lb = 0
        state.sub = 0
        return state

    def is_dominated_by(self, other):
        if other.node == self.node and other.path == self.path:
            return True
        return False

    def __str__(self):
        return f'({self.q}, {self.node})'

    def __repr__(self):
        return f'({self.q}, {self.node})'


class Label:
    """
    Clase utilizada para representar a las etiquetas en el algoritmo que busca caminos con el menor costo reducido.
    Cada etiqueta tiene los siguientes atributos:
    node : último nodo del camino
    q : carga acumulada a lo largo del camino
    pred : nodo predecesor a 'node' en el camino
    path : camino
    reduced_cost : costo reducido de la ruta que resulta de agregarle al camino el arco (v,0)
    """

    def __init__(self, q, j):
        self.node = j
        self.q = q
        self.pred = None
        self.path = []
        self.reduced_cost = inf

    @classmethod
    def warehouse_label(cls):
        label = cls(0, 0)
        label.path = [0]
        label.reduced_cost = 0
        return label

    def __str__(self):
        return f'L({self.q}, {self.node})'

    def __repr__(self):
        return f'L({self.q}. {self.node})'