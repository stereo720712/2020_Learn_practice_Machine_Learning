import MLDemo1CL as mld




if __name__ == '__main__':

    mlDemo = mld.MLDemo1CL()
    #1
    mlDemo.loadData()
    #2
    #mlDemo.showSourceData_nb()
    #3
    #mlDemo.showSourceDataCountGraph()
    #error mlDemo.showWordFashionGraph()

    #4  data process
    mlDemo.data_process()

    #model training
   # mlDemo.model_train()

    # model test for confusion matrix
    mlDemo.model_test()
    mlDemo.show_confusion_matrix_demo()

    #predict
    #mlDemo.predict('verify.txt')
