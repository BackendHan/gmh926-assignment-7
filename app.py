from flask import Flask, render_template, request, session, url_for
import numpy as np
import matplotlib
from scipy.stats import t
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key

def generate_data(N, mu, beta0, beta1, sigma2, S):
    X = np.random.rand(N)
    error = np.random.normal(0, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + mu + error

    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color="blue", label="Data")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label="Fitted Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plot1_path = os.path.join("static", "plot1.png")
    plt.savefig(plot1_path)
    plt.close()

    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.rand(N)
        error_sim = np.random.normal(0, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + mu + error_sim
        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    plt.figure(figsize=(8, 6))
    plt.hist(slopes, bins=20, alpha=0.7, color="blue")
    plt.xlabel("Slope")
    plt.ylabel("Frequency")
    plt.title("Histogram of Slopes from Simulations")
    plot2_path = os.path.join("static", "plot2.png")
    plt.savefig(plot2_path)
    plt.close()

    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slopes,
        intercepts,
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    session.clear()
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    if test_type == "!=":
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))
    elif test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    else:
        p_value = np.mean(simulated_stats <= observed_stat)

    fun_message = "Extremely significant result!" if p_value <= 0.0001 else None

    plt.figure(figsize=(8, 6))
    plt.hist(simulated_stats, bins=20, color="lightgray")
    plt.axvline(observed_stat, color="red", linestyle="--", label="Observed Statistic")
    plt.axvline(hypothesized_value, color="green", linestyle="-", label="Hypothesized Value")
    plt.xlabel(parameter.capitalize())
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Simulated {parameter.capitalize()}s")
    plt.legend()
    plot3_path = os.path.join("static", "plot3.png")
    plt.savefig(plot3_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    try:
        N = int(session.get("N"))
        mu = float(session.get("mu"))
        sigma2 = float(session.get("sigma2"))
        beta0 = float(session.get("beta0"))
        beta1 = float(session.get("beta1"))
        S = int(session.get("S"))

        parameter = request.form.get("parameter")
        confidence_level = float(request.form.get("confidence_level"))

        if parameter == "slope":
            estimates = np.array(session.get("slopes"))
            observed_stat = float(session.get("slope"))
            true_param = beta1
        else:
            estimates = np.array(session.get("intercepts"))
            observed_stat = float(session.get("intercept"))
            true_param = beta0

        alpha = 1 - confidence_level / 100
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        ci_lower = np.percentile(estimates, lower_percentile)
        ci_upper = np.percentile(estimates, upper_percentile)
        includes_true = ci_lower <= true_param <= ci_upper

        plt.figure(figsize=(8, 6))
        plt.scatter(estimates, [0]*len(estimates), color="gray", alpha=0.5, label="Simulated Estimates")
        plt.axvline(true_param, color="green", linestyle="--", linewidth=2, label="True Parameter")
        plt.plot([ci_lower, ci_upper], [0, 0], color="blue", linewidth=5, label=f"{confidence_level}% Confidence Interval")
        plt.plot(np.mean(estimates), 0, 'o', color="red", markersize=10, label="Mean Estimate")
        plt.xlabel(f"{parameter.capitalize()} Estimate")
        plt.yticks([])
        plt.title(f"{confidence_level}% Confidence Interval for {parameter.capitalize()}")
        plt.legend()
        plot4_path = os.path.join("static", "plot4.png")
        plt.savefig(plot4_path, bbox_inches='tight')
        plt.close()

        return render_template(
            "index.html",
            plot1="static/plot1.png",
            plot2="static/plot2.png",
            plot4=plot4_path,
            parameter=parameter,
            confidence_level=confidence_level,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            includes_true=includes_true,
            observed_stat=observed_stat,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    except Exception as e:
        print("An error occurred:", e)
        return "An error occurred while calculating the confidence interval."

if __name__ == "__main__":
    app.run(debug=True)
