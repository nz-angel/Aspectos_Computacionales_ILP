from docplex.mp.model import Model
from collections import Counter


class UnfeasibilityError(Exception):
    pass


def number_of_visits(travel, i):
    """ Cuenta la cantidad de veces que una instancia de Path o de Route visita un nodo i """
    return travel.sequence.count(i)


def write_clp_model(routes, instance):
    """
    Escribe el modelo CLP a partir de las rutas en la lista 'routes' (cada instancia de Route tiene como atributo el
    tipo de vehículo al que le corresponde).
    :param routes: lista con las rutas que determinarán las columnas de CLP
    :type routes: list[Route]
    :param instance: datos del problema
    :type instance: Instance
    :return: modelo CLP
    :rtype: Model
    """
    clients = set(instance.demand).difference({0})

    model = Model('CLP')

    # Se definen las variables utilizando la misma notación del paper
    xidx = [(idx, route.vehicle) for idx, route in enumerate(routes)]
    x = model.continuous_var_dict(xidx, lb=0, name='X')
    y = model.continuous_var_dict(instance.cap, lb=0, name='Y')

    model.xvars = x

    # Agregamos las restricciones
    for i in clients:
        model.add_constraint(model.sum(x[r, k] * number_of_visits(routes[r], i) for r, k in xidx) >= 1,
                             ctname=f'Cobertura de cliente {i}')

    for k in instance.cap:
        model.add_constraint(model.sum(x[t] for t in xidx if t[1] == k) == y[k],
                             ctname=f'Cant. de vehículos de tipo {k} utilizados')

    # Escribimos la función objetivo
    model.minimize(model.sum(routes[r].cost * x[r, k] for r, k in xidx))

    model.parameters.preprocessing.presolve(0)
    return model


def solve_clp_model(model, instance, routes):
    """
    Resuelve CLP.
    :param model: modelo CLP
    :type model: Model
    :param instance: datos del problema
    :type instance: Instance
    :param routes: lista con las rutas correspondientes a las columnas de CLP
    :type routes: list[Route]
    :return: devuelve los valores óptimos de las variables del dual y, si la solución es entera, una lista con las
    rutas de routes que forman parte de ella.
    :rtype: dict, dict, list[Route]
    """
    clients = set(instance.demand).difference({0})

    sol = model.solve(log_output=False)
    if sol is not None:
        # Recuperamos el valor de las variables del dual
        pi = {}
        lamb = {}
        for i in clients:
            pi[i] = model.get_constraint_by_name(f'Cobertura de cliente {i}').dual_value
        for k in instance.cap:
            lamb[k] = model.get_constraint_by_name(f'Cant. de vehículos de tipo {k} utilizados').dual_value

        solution = []
        if solution_is_integer(sol):
            xsol = sol.get_value_dict(model.xvars, keep_zeros=False)
            for (r, k) in xsol:
                solution.append((routes[r], k))
        return pi, lamb, solution
    else:
        raise UnfeasibilityError


def solve_cp_model(routes, instance):
    """
    Plantea y resuelve el modelo CP.
    :param routes: lista con las rutas correspondientes a las columnas de CLP
    :type routes: list[Route]
    :param instance: datos del problema
    :type instance: Instance
    :return: una lista con las rutas de routes que forman parte de la solución de CP.
    :rtype: list[Route]
    """
    clients = set(instance.demand).difference({0})

    model = Model('CP')

    # Se definen las variables utilizando la misma notación del paper
    xidx = [(idx, route.vehicle) for idx, route in enumerate(routes)]
    x = model.binary_var_dict(xidx, name='X')
    y = model.integer_var_dict(instance.cap, lb=0, name='Y')

    # Agregamos las restricciones
    for i in clients:
        model.add_constraint(model.sum(x[r, k]*number_of_visits(routes[r], i) for r, k in xidx) >= 1,
                             ctname=f'Cobertura de cliente {i}')

    for k in instance.cap:
        model.add_constraint(model.sum(x[t] for t in xidx if t[1] == k) == y[k],
                             ctname=f'Cant. de vehículos de tipo {k} utilizados')

    # Escribimos la función objetivo
    model.minimize(model.sum(routes[r].cost*x[r, k] for r, k in xidx))

    model.parameters.timelimit(30*60)
    sol = model.solve(log_output=False)
    if sol is not None:
        solution = []
        xsol = sol.get_value_dict(x, False)
        for (r, k) in xsol:
            solution.append(routes[r])
        return solution
    else:
        raise UnfeasibilityError


def solution_is_integer(solution):
    return all(s.is_integer() for s in solution.as_dict().values())


def vehicle_preprocessing_model(k, instance):
    """
    Plantea y resuelve el modelo de programación lineal entera que proponen los autores para eliminar tipos de
    vehículos que no pueden formar parte de la solución óptima.
    :param k: tipo de vehículo del cual se debe utilizar al menos uno.
    :type k: int
    :param instance: datos del problema
    :type instance: Instance
    :return: cota inferior del problema HVRP correspondiente a la instancia si se agrega la restricción de utilizar
    al menos un vehículo de tipo k.
    :rtype: float
    """
    model = Model()

    # Agregamos las variables
    y = model.integer_var_dict(instance.cap)

    # Agregamos las restricciones
    model.add_constraint(model.sum(instance.cap[t]*y[t] for t in instance.cap) >= model.sum(instance.demand.values()))
    model.add_constraint(y[k] >= 1)

    # Funcion objetivo
    model.minimize(model.sum(instance.fixed_cost[t]*y[t] for t in instance.cap))

    sol = model.solve()
    return sol.objective_value
