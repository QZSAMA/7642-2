import numpy as np
import mlrose_hiive as mlrose

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from datetime import datetime
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def fill_tail(fitness_values,count):
    current_size=len(fitness_values)
    if current_size<count:
        return np.array(list(fitness_values)+[fitness_values[-1]]*(count-current_size))
    else:
        return fitness_values
        
def run_problem_algorithm(problem,
                          algorithm,
                        #   gtid=903789757,
                          num_runs=30,
                          max_attempts=100,
                          max_iters=1000):
    # results = []
    fitness_curves=[]
    for _ in range(num_runs):
        best_state, best_fitness, fitness_curve = algorithm(problem, 
                                                           max_attempts=max_attempts, 
                                                           max_iters=max_iters,
                                                           curve=True)
        # results.append(best_fitness)
        fitness_values = fill_tail(fitness_curve[:, 0],max_iters)
        # fitness_values = fitness_curve[:, 0]
        fitness_curves.append(fitness_values)
    return fitness_curves

def draw_curve(fitness_curves,name,color):
    mean_curve = np.mean(fitness_curves, axis=0)
    print("Mean curve fitness :",mean_curve[-1])
    std_curve = np.std(fitness_curves, axis=0)
    print("Std curve fitness :",std_curve[-1])
    x = np.arange(0, mean_curve.shape[0])
    plt.plot(x, mean_curve, color=color, label='Mean Curve of '+name)
    plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2, label='Standard Deviation of '+name)
    return



def run_problem(problem,problem_name,num_runs=20,max_attempts=100,max_iters=300):
    algorithms = {
    'Randomized Hill Climbing': (mlrose.random_hill_climb,'red'),
    'Simulated Annealing': (mlrose.simulated_annealing,'blue'),
    'Genetic Algorithm': (mlrose.genetic_alg,'orange'),
    'MIMIC':(mlrose.mimic,'green')
    }
    # algorithms = {
    # 'Randomized Hill Climbing': (mlrose.random_hill_climb,'red'),
    # 'Simulated Annealing': (mlrose.simulated_annealing,'blue')
    # }
    plt.figure()
    for algo_name in algorithms:
        print(algo_name)
        begin_time=datetime.now()   
        algorithm,color=algorithms[algo_name]
        fitness_curves=run_problem_algorithm(problem,
                              algorithm,
                              num_runs=num_runs,
                              max_attempts=max_attempts,
                              max_iters=max_iters)
        end_time=datetime.now()
        total_time = end_time - begin_time
        print(algo_name+" Time Cost ("+str(num_runs)+"runs) :\t"+str(total_time)+"\tAverage time cost : "+str(total_time/num_runs))
        draw_curve(fitness_curves,algo_name,color)
    plt.title('Fitness Curve of Optimization Algorithms of problem '+problem_name)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    # Add a legend
    plt.legend()

    # Show the plot
    # plt.show()
    plt.savefig(problem_name+'.png')

def generate_random_coords(num_points, x_max, y_max,gtid):
    np.random.seed(gtid)  
    coords = np.random.rand(num_points, 2) * [x_max, y_max]
    tuple_coords = [tuple(row) for row in coords]
    return tuple_coords

def neural_network(X_train, X_test, y_train, y_test,graph=False,max_iter=300,size=70):
    mlp_clf = MLPClassifier(hidden_layer_sizes=(size, size), max_iter=max_iter)
    begin_time=datetime.now()
    mlp_clf.fit(X_train, y_train)
    end_time=datetime.now()
    # y_pred = mlp_clf.predict(X_test)
    train_score = mlp_clf.score(X_train, y_train)
    test_score = mlp_clf.score(X_test, y_test)
    return train_score,test_score,mlp_clf,end_time-begin_time

def run_sk_nn(X_train, X_test, y_train, y_test):
    print('- - - Neural Network (scikit learn with adam solver) - - -')
    train_score_nn,test_score_nn,mlp_clf,train_time = neural_network(X_train, X_test, y_train, y_test,graph=False)
    begin_time=datetime.now()    
    pred_y=mlp_clf.predict(X_test)
    end_time=datetime.now()
    print("Time cost - Train  :",str(train_time),"\tTest :",str(end_time-begin_time))
    print('Score - Train : ',train_score_nn,'\tTest : ',test_score_nn)
    plot(mlp_clf.loss_curve_,"Sklearn MLP")
    
    print("Current Loss: ",mlp_clf.loss_)
    return mlp_clf

def plot(curve,name):
    plt.figure(name )
    iters=np.arange(1, len(curve)+1,1)
    plt.xlabel("Iterations")
    plt.ylabel("Loss value")
    plt.title("Loss value vs Iterations for Neural Network with "+str(name))
    plt.plot(iters, curve, marker="o",markersize=2, label=str(name))
    plt.savefig(name+".png")

def run_ga_nn(X_train, X_test, y_train, y_test,
                hidden_node_count=70,
                learning_rate=0.1,
                max_iters=500,
              pop_size=300,
              mutation_prob=0.005,
              gtid=903789757):
    print('- - - Neural Network (Genetic Algorithm)- - -')
    ga_nn = mlrose.NeuralNetwork(
        hidden_nodes=[hidden_node_count],
        activation='relu',
        algorithm='genetic_alg',
        max_iters=max_iters,
        is_classifier =False,
        learning_rate=learning_rate,
        pop_size=pop_size,
        mutation_prob=mutation_prob,
        # max_attempts=15,
        random_state=gtid,
        curve=True,
        # early_stopping =True
    )
    begin_time=datetime.now()
    ga_nn.fit(X_train, y_train)
    end_time=datetime.now()
    print("Training Complete with time: ",end_time-begin_time)
    plot(ga_nn.fitness_curve[:,0],"Genetic Algorithm")
    print("Current Loss: ",ga_nn.loss )
    return ga_nn    
    

def run_sa_nn(X_train, X_test, y_train, y_test,
                hidden_node_count=70,
                learning_rate=0.1,
                max_iters=200000,
                gtid=903789757):
    print('- - - Neural Network (Simulated Annealing)- - -')
    sa_nn = mlrose.NeuralNetwork(
        hidden_nodes=[hidden_node_count],
        activation='relu',
        algorithm='simulated_annealing',
        max_iters=max_iters,
        is_classifier =False,
        learning_rate=learning_rate,
        # max_attempts=150,
        random_state=gtid,
        curve=True,
        # early_stopping =True
    )
    begin_time=datetime.now()
    sa_nn.fit(X_train, y_train)
    end_time=datetime.now()
    print("Training Complete with time: ",end_time-begin_time)
    print("Current Loss: ",sa_nn.loss )
    plot(sa_nn.fitness_curve[:,0],"Simulated Annealing")
    return sa_nn

def run_rhc_nn(X_train, X_test, y_train, y_test,
                hidden_node_count=70,
                learning_rate=0.1,
                max_iters=200000,
                gtid=903789757):
    print('- - - Neural Network (Random Hill Climbing)- - -')
    rhc_nn = mlrose.NeuralNetwork(
        hidden_nodes=[hidden_node_count],
        activation='relu',
        algorithm='random_hill_climb',
        max_iters=max_iters,
        is_classifier =False,
        learning_rate=learning_rate,
        # max_attempts=150,
        random_state=gtid,
        curve=True,
        restarts=0,
        # early_stopping =True
    )
    begin_time=datetime.now()
    rhc_nn.fit(X_train, y_train)
    end_time=datetime.now()
    print("Training Complete with time: ",end_time-begin_time)
    print("Current Loss: ",rhc_nn.loss )
    plot(rhc_nn.fitness_curve[:,0],"Random Hill Climbing")
    return rhc_nn
# def run_gd_nn(X_train, X_test, y_train, y_test,gtid):
#     rhc_nn = mlrose.NeuralNetwork(
#         hidden_nodes=[50],
#         activation='relu',
#         algorithm='gradient_descent',
#         max_iters=150000,
#         is_classifier =False,
#         learning_rate=0.01,
#         # max_attempts=150,
#         random_state=gtid,
#         curve=True,
#     #     early_stopping =True
#     )
#     begin_time=datetime.now()
#     rhc_nn.fit(X_train, y_train)
#     end_time=datetime.now()
#     print("Training Complete with time: ",end_time-begin_time)
#     plot(rhc_nn.fitness_curve[:,0],"Gradient Descent")

def run_nns(data,gtid):
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=gtid)
    sk_nn=run_sk_nn(X_train, X_test, y_train, y_test)
    # small
    hidden_node_count=70
    learning_rate=0.1
    max_iters=200
    # big
    hidden_node_count=70
    learning_rate=0.1
    max_iters=10000
    rhc_nn=run_rhc_nn(X_train, X_test, y_train, y_test,hidden_node_count,learning_rate,max_iters,gtid)
    sa_nn=run_sa_nn(X_train, X_test, y_train, y_test,hidden_node_count,learning_rate,max_iters,gtid)
    ga_nn=run_ga_nn(X_train, X_test, y_train, y_test,hidden_node_count,learning_rate,max_iters)
    # run_gd_nn(X_train, X_test, y_train, y_test,gtid)
    print(1)



# def run_problem_size(min_n,max_n):


def main():

    gtid=903789757

    num_runs= 20
    max_attempts=100
    max_iters=250


    # TSP
    num_cities = 10
    x_max = 10
    y_max = 10
    
    print('Processing Traveling Salesman Problem')
    coords_list = generate_random_coords(num_cities, x_max, y_max,gtid)
    print('TSP Locations Generated : ',coords_list)
    fitness_dists = mlrose.TravellingSales(coords=coords_list)
    problem_TSP = mlrose.TSPOpt(length=len(coords_list), fitness_fn=fitness_dists, maximize=False)
    run_problem(problem_TSP,'Traveling Salesman',num_runs,max_attempts,max_iters)
    print('Processing N-Queen Problem (N=5)')
    problem_nqueen=mlrose.DiscreteOpt(length=5, fitness_fn = mlrose.Queens(), maximize=False, max_val=5)
    run_problem(problem_nqueen,'N-Queen',num_runs,max_attempts,max_iters)



    digits = load_digits()
    run_nns(digits,gtid)

    # print(1)

if __name__ == '__main__':
    main()