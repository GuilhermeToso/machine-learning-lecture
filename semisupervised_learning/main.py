from semisupervised_learning import SSLFramework

ssl = SSLFramework()
ssl.load_ssl_model()
ssl.evaluate()
ssl.plot_train_data()
ssl.plot_decision_boundary()
ssl.plot_decision_boundary("test")