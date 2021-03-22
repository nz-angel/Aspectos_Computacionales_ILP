import os
from collections import deque
from itertools import chain
from math import isclose, inf

from classes import Path, Route, Label, State, ProblemInstance
from models import vehicle_preprocessing_model, write_clp_model, solve_clp_model, solve_cp_model


def initialize_columns(instance):
    """ Se inicializan las columnas: cada cliente es atendido exclusivamente por un vehículo con la capacidad mínima
    entre todos los vehículos que pueden satisfacerlo. """
    routes = []

    clients = set(instance.demand)
    clients.remove(0)

    for i in clients:
        # Entre los vehículos que pueden satisface la demanda de i, elegimos al de menor capacidad
        k = min((v for v, c in instance.cap.items() if c >= instance.demand[i]), key=lambda v: instance.cap[v])
        path = Path(k)
        path.add_node(i, instance)
        routes.append(Route.from_path(path, instance))

    return routes


def solve_2cycle_free_SPbar(pi, lamb, instance):
    """
    Se resuelve el problema \bar{SP}, que es el problema auxiliar para resolver CLP.
    :param pi: diccionario con los valores óptimos de las variables del dual correspondiente al primer conjunto de
    restricciones de CLP
    :type pi: dict
    :param lamb: diccionario con los valores óptimos de las variables del dual correspondiente al primer conjunto de
    restricciones de CLP
    :type lamb: dict
    :param instance: datos del problema
    :type instance: Instance
    :return: tupla con la ruta correspondiente a la potencial nueva columna de CLP y su costo reducido
    :rtype: Route, float
    """
    min_red_cost = 1e9
    best_route = None
    for k in instance.cap:
        route, red_cost = solve_SPk(pi, lamb, k, instance)
        if abs(red_cost) < 1e-5:    # Esto es necesario para evitar incluir columnas por error numérico
            red_cost = 0
        if red_cost < min_red_cost:
            min_red_cost = red_cost
            best_route = route
    return best_route, min_red_cost


def solve_SPk(pi, lamb, k, instance):
    """
    Se resulve el problema \bar{SP} restringido al tipo de vehículo k. Se utiliza el procedimiento de etiquetas
    detallada en la tercera sección del informe.
    :param pi: diccionario con los valores óptimos de las variables del dual correspondiente al primer conjunto de
    restricciones de CLP
    :type pi: dict
    :param lamb: diccionario con los valores óptimos de las variables del dual correspondiente al primer conjunto de
    restricciones de CLP
    :type lamb: dict
    :param k: tipo de vehículo
    :type k: int
    :param instance: datos del problema
    :type instance: Instance
    :return: ruta con menor costo reducido para el tipo de vehículo k y su costo reducido
    :rtype: Route, float
    """
    pi[0] = lamb[k]

    bk = instance.cap[k]

    # Removemos a los clientes cuya demanda supera a bk
    nodes = set(filter(lambda c: instance.demand[c] <= bk, instance.demand))
    clients = nodes.difference({0})

    # Se define la matriz de costos del subgrafo \bar{G}_k
    cost_matrix = make_cost_matrix(pi, k, instance)

    # Se inicializa el estado del depósito
    warehouse_label = Label.warehouse_label()

    # Se crean los conjuntos de estados cuya cota superior coincide con su cota inferior
    P = {j: [] for j in clients}
    P[0] = [warehouse_label]

    L = deque([warehouse_label])
    while L:
        label = L.popleft()

        for j in filter(lambda c: label.q + instance.demand[c] <= bk and c != label.pred and c != label.node, clients):
            new_label = Label(label.q + instance.demand[j], j)
            new_label.path = label.path + [j]
            new_label.pred = label.node
            new_label.reduced_cost = get_reduced_cost(new_label, cost_matrix, instance, k)

            dominates = []
            dominated = False

            # Se chequea si la nueva etiqueta no está dominada
            for s in P[j]:
                if s.reduced_cost > new_label.reduced_cost:
                    dominates.append(s)
                elif s.reduced_cost < new_label.reduced_cost:
                    dominated = True
                    break
            if not dominated:
                # Se encola la nueva etiqueta
                L.append(new_label)
                P[j].append(new_label)

                # Se eliminan todas las etiquetas dominadas por la nueva
                for s in dominates:
                    P[j].remove(s)
                    try:
                        L.remove(s)
                    except ValueError:
                        pass

    best_state = min((s for s in chain.from_iterable(P.values())), key=lambda s: getattr(s, 'reduced_cost'))
    best_path = Path.from_sequence(best_state.path, instance, k)
    return Route.from_path(best_path, instance), get_reduced_cost(best_state, cost_matrix, instance, k)


def pulling_algorithm(pi, lamb, k, instance):
    """
    Método para resolver \bar{SP} restringido al tipo de vehículo k desarrollado por Desrochers. No utilizado para
    resolver las instancias.
    :param pi: diccionario con los valores óptimos de las variables del dual correspondiente al primer conjunto de
    restricciones de CLP
    :type pi: dict
    :param lamb: diccionario con los valores óptimos de las variables del dual correspondiente al primer conjunto de
    restricciones de CLP
    :type lamb: dict
    :param k: tipo de vehículo
    :type k: int
    :param instance: datos del problema
    :type instance: Instance
    :return: ruta con menor costo reducido para el tipo de vehículo k y su costo reducido
    :rtype: Route, float
    """
    pi[0] = lamb[k]

    bk = instance.cap[k]

    # Removemos a los clientes cuya demanda supera a bk
    nodes = set(filter(lambda c: instance.demand[c] <= bk, instance.demand))
    clients = nodes.difference({0})

    # Se define la matriz de costos del subgrafo \bar{G}_k
    cost_matrix = make_cost_matrix(pi, k, instance)

    # Se inicilizan los estados de los nodos correspondientes a los clientes
    states = [State(q, j) for j in clients for q in range(instance.demand[j], bk + 1)]

    # Se inicializa el estado del depósito
    depo_state = State.warehouse_state()
    states.append(depo_state)

    # Se crean los conjuntos de estados cuya cota superior coincide con su cota inferior
    P = {j: [] for j in clients}
    P[0] = [depo_state]

    # Consideramos el conjunto W de estados cuyas cotas no coinciden
    W = deque(sorted([s for s in states if s.ub != s.lb], key=lambda s: (s.q, s.node)))

    best_state = None
    best_rc = 1e6

    while W:
        # Entre los estados de cada j cuya cotas no coinciden, elegimos el que tiene menor q. Ante empates, elegimos el
        # que tiene menor j
        state = W.popleft()

        # Actualizamos la cota superior del estado. Primero calculamos el estado previo a (q,j) para el cual se realiza
        # el mínimo
        demand_bound = state.q - instance.demand[state.node]
        candidate_states = [s for s in states if s.node != state.node and s.q <= demand_bound]
        candidate_phis = [phi(s, state.node, cost_matrix) for s in candidate_states]
        prev_state, prev_phi = get_prev_state(candidate_states, candidate_phis)
        state.pred = prev_state.node
        state.ub = prev_phi
        state.path = prev_state.path + [state.node]

        # Se actualiza la cota superior del segundo mejor camino. Si el conjunto sobre el que se toma el mínimo es
        # vacío, la cota no se altera
        state.sub = get_secondary_ub(candidate_states, candidate_phis, state.pred)

        # Se actualiza la cota inferior del estado
        state.lb = min(map(lambda s: s.lb + cost_matrix[s.node][state.node], candidate_states))

        # Actualizamos Pj de ser necesario:
        if isclose(state.lb, state.ub):
            P[state.node].append(state)
            state_rc = get_reduced_cost(state, cost_matrix, instance, k)
            if state_rc < best_rc:
                best_rc = state_rc
                best_state = state

    best_path = Path.from_sequence(best_state.path, instance, k)
    return Route.from_path(best_path, instance), get_reduced_cost(best_state, cost_matrix, instance, k)


def make_cost_matrix(pi, k, instance):
    """
    Inicializa la matriz de costos del grafo correspondiente a SP(k).
    :param pi: diccionario con los valores óptimos de las variables del dual correspondiente al primer conjunto de
    restricciones de CLP
    :type pi: dict
    :param k: tipo de vehículo
    :type k: int
    :param instance: datos del problema
    :type instance: Instance
    :return: matriz con los costos de los arcos del grafo correspondiente a SP(k)
    :rtype: dict[int, dict]
    """
    cost_matrix = {i: {} for i in instance.demand}
    for i in instance.demand:
        for j in filter(lambda j: j != i, instance.demand):
            cost_matrix[i][j] = instance.arc_cost(k, i, j) - pi[j]
        cost_matrix[i][i] = 0
    return cost_matrix


def phi(state, j, cost_matrix):
    """ Función phi para el método de Desrochers """
    if state.pred != j:
        return state.ub + cost_matrix[state.node][j]
    else:
        return state.sub + cost_matrix[state.node][j]


def get_prev_state(candidate_states, candidate_phis):
    """ Efectúa el cálculo del mínimo entre los phi de los demás nodos, para el cálculo de F_ k^1"""
    best_state = None
    best_phi = 1e6
    for idx, phi in enumerate(candidate_phis):
        if phi < best_phi:
            best_phi = phi
            best_state = candidate_states[idx]
    return best_state, best_phi


def get_secondary_ub(candidate_states, candidate_phis, pred):
    """ Efectúa el cálculo del mínimo entre los phi de los demás nodos, para el cálculo de F_ k^2"""
    best_phi = inf
    for idx, phi in enumerate(candidate_phis):
        if candidate_states[idx].node != pred and phi < best_phi:
            best_phi = phi
    return best_phi


def get_reduced_cost(state, cost_matrix, instance, k):
    """ Calcula el costo reducido del State o Label, según el método que se esté utilizando """
    if isinstance(state, State):
        return state.ub + cost_matrix[state.node][0] + instance.fixed_cost[k]
    elif isinstance(state, Label):
        return sum(cost_matrix[i][j] for i, j in zip(state.path[:-1], state.path[1:])) + (cost_matrix[state.node][0] +
                                                                                          instance.fixed_cost[k])


def vehicle_type_preprocessing(routes, instance):
    """ Efectúa el preprocesamiento para eliminar tipos de vehículos que no pueden formar parte de la solución
    óptima.
    :param routes: lista de las rutas iniciales, donde cada cliente es exclusivamente atendido por un vehículo con
    capacidad mínima para satisfacer su demanda.
    :type routes: list[Route]
    :param instance: datos del problema
    :type instance: Instance
    """
    upper_bound = sum(route.cost for route in routes)
    k = max(instance.cap)

    while k > 0:
        lower_bound = vehicle_preprocessing_model(k, instance)
        if lower_bound > upper_bound:
            instance.cap.pop(k)
            instance.fixed_cost.pop(k)
            instance.var_cost.pop(k)
        k -= 1


def print_solution(solution):
    for route in solution:
        print(f'Vehículo de tipo {route.vehicle} recorre ruta {route.sequence} con costo '
              f'{round(route.cost,3)} y demanda total {route.demand}')
    print(f'Costo total: {sum(route.cost for route in solution)}')


def cg_algorithm(instance_filename):
    """
    Aplica el algoritmo de los autores a la instancia indicada. Imprime en pantalla el resultado.
    :param instance_filename: nombre del archivo en la carpeta Instancias que corresponde a la instancia a resolver.
    :type instance_filename: str
    """

    instance = ProblemInstance(os.path.join('Instancias', instance_filename))

    # Construir CLP inicial:
    routes = initialize_columns(instance)
    clp = write_clp_model(routes, instance)

    # Llevamos a cabo el preprocesamiento para ver si se puede eliminar algún tipo de vehículo
    vehicle_type_preprocessing(routes, instance)

    print('Generando columnas')
    new_cols = 0
    while True:

        # Resolver el CLP. devolver el valor óptimo de las variables del primal y del dual
        pi, lamb, solution = solve_clp_model(clp, instance, routes)

        # Resolver 2-cycle-free SP: devolver ruta y costo reducido
        route, reduced_cost = solve_2cycle_free_SPbar(pi, lamb, instance)

        # Si el costo reducido es negativo, se agrega la nueva columna correspondiente a la ruta hallada
        if reduced_cost < 0:
            routes.append(route)
            clp = write_clp_model(routes, instance)
            new_cols += 1
        else:
            break

    print(f'Se generaron {new_cols} columnas')

    # Si la solución del último CLP no es entera, plantear y resolver CP con las rutas de P'
    if not solution:
        print(f'Resolviendo CP')
        solution = solve_cp_model(routes, instance)

    print_solution(solution)
