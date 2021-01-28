import pickle

def run_heatmap_experiment(ano_indicator=True, in_bag=True, n_exp=1):
    
    rng = np.random.RandomState(seed_val)
    
    #Keep track of the negative values
    neg_dict = {}
    neg_dict_l = []
    
    #Define the starting dataset
    min_val = -2
    max_val = 2
    X = 0.3 * rng.randn(100, 2)
    initial_dataset = np.r_[X + max_val, X + min_val]
    colors = ["c"] * 200
    if ano_indicator:
        ano_points = rng.uniform(low=-4, high=4, size=(20, 2))
        initial_dataset = np.r_[initial_dataset, ano_points]
        colors.extend(["tab:pink"]*20)
    
    #Create meshgrid of possible combinations
    nx = 100
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, nx)
    
    aws_matrix_final = np.zeros(shape=(len(x), len(y)))
    aws_matrix_temp = np.zeros(shape=(len(x), len(y)))
    aws_matrix_collection = np.zeros(shape=(n_exp, len(x), len(y)))
    
    diffi_matrix_final = np.zeros(shape=(len(x), len(y)))
    diffi_matrix_temp = np.zeros(shape=(len(x), len(y)))
    diffi_matrix_collection = np.zeros(shape=(n_exp, len(x), len(y)))
    
    for i in range(n_exp):
      
        #Fit an IF model to this starting dataset
        clf = IsolationForest(n_estimators=100, random_state=rng, max_samples=256)
        clf.fit(initial_dataset)
        
        #Loop through possible coordinates starting at normal
        for j, element in enumerate(list(itertools.product(x, y))):
            
            x_val = element[0]
            y_val = element[1]
            
            #If ano, fit the IF to the new dataset with the point
            if in_bag:
                del clf
                clf = IsolationForest(n_estimators=100, random_state=rng, max_samples=256)
                clf.fit(np.r_[initial_dataset, np.array([[x_val, y_val]])])
                    
            #Generate explanations - AWS, DIFFI
            aws_unnorm, _ = point_aws_explanation(clf, np.array([x_val, y_val]), False, 'clement')
            diffi_unnorm, _ = interp.local_diffi(clf, np.array([x_val, y_val]))
        
            #Check whether the AWS vector has any negative value
            if (aws_unnorm < 0).any():
                neg_dict[(x_val, y_val)] = 1
                aws_unnorm[aws_unnorm < 0] = 0
            
            #Process the explanation vectors
            aws_norm = aws_unnorm/np.sum(aws_unnorm)
            diffi_norm = diffi_unnorm/np.sum(diffi_unnorm)
    
            #Append to some array-like object for both AWS and DIFFI
            x_index = j % nx
            y_index = j // nx
            aws_matrix_temp[x_index, y_index] = aws_norm[0]
            diffi_matrix_temp[x_index, y_index] = diffi_norm[0]
            
        neg_dict_l.append(neg_dict)
        aws_matrix_collection[i] = aws_matrix_temp
        diffi_matrix_collection[i] = diffi_matrix_temp
    
    #Append to some array-like object for both AWS and DIFFI
    aws_matrix_final = np.mean(aws_matrix_collection, axis=0)
    diffi_matrix_final = np.mean(diffi_matrix_collection, axis=0)
        
    #Plotting function for both AWS & DIFFI arrays
    #x and y should be X1 and X2. cmap should be between 0(Red, X1=0, X2=1) and 1 (Blue.X1=1, X2=0)
    #x,y discretized to meshgrid
    # plot the line, the samples, and the nearest vectors to the plane
    xx, yy = np.meshgrid(x, y)
    aws_matrix_final = aws_matrix_final.reshape(xx.shape)
    diffi_matrix_final = diffi_matrix_final.reshape(xx.shape)
    
    aws_save_name = "AWS_" + "ano_" + str(ano_indicator) + "_inbag" + str(in_bag) + ".pdf"
    diffi_save_name = "DIFFI_" + "ano_" + str(ano_indicator) + "_inbag" + str(in_bag) + ".pdf"
    
    os.chdir(dir_path)
    
    plt.close()
    plt.style.use('seaborn')
    sns.set_context("talk", font_scale=1.5)
    plt.contourf(xx, yy, aws_matrix_final, cmap=plt.cm.RdBu, vmin=0, vmax=1)
    m = plt.cm.ScalarMappable(cmap=plt.cm.RdBu)
    m.set_array(aws_matrix_final)
    m.set_clim(0., 1.)
    plt.colorbar(m, boundaries=np.linspace(0, 1, 11))
    plt.scatter(initial_dataset[:, 0], initial_dataset[:, 1], c=colors, s=80, edgecolor='k')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.tick_params(top=False, bottom=False, left=False, right=False,
               labelleft=False, labelbottom=False)
    plt.show()
    if savefiles:
      plt.savefig('contour_' + aws_save_name, bbox_inches='tight')
    
    plt.close()
    plt.style.use('seaborn')
    sns.set_context("talk", font_scale=1.5)
    plt.contourf(xx, yy, diffi_matrix_final, cmap=plt.cm.RdBu, vmin=0, vmax=1)
    m = plt.cm.ScalarMappable(cmap=plt.cm.RdBu)
    m.set_array(diffi_matrix_final)
    m.set_clim(0., 1.)
    plt.colorbar(m, boundaries=np.linspace(0, 1, 11))
    plt.scatter(initial_dataset[:, 0], initial_dataset[:, 1], c=colors, s=80, edgecolor='k')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.tick_params(top=False, bottom=False, left=False, right=False,
               labelleft=False, labelbottom=False)
    plt.show()
    if savefiles:
      plt.savefig('contour_' + diffi_save_name, bbox_inches='tight') 
    
    return neg_dict_l, aws_matrix_final, diffi_matrix_final

