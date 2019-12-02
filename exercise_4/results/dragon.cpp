#define _USE_MATH_DEFINES
#include "utils/io.h"
#include "utils/points.h"

#include "ceres/ceres.h"
#include <cmath>

template <typename Type, int Size> using Vector = Eigen::Matrix<Type, Size, 1>;

// DONE: Implement the cost function
struct RegistrationCostFunction
{
    explicit RegistrationCostFunction(const Point2D & input, const Point2D & target, const Weight & weight) :
        input(input), target(target), weight(weight) {};

    template<typename T>
    bool operator()(const T* const deg, const T* const tx, const T* const ty, T* residual) const {
        // DONE: Implement the cost function
        Eigen::Rotation2D<T> rotation(deg[0]);
        Eigen::Translation<T, 2> translation(tx[0], ty[0]);

        Vector<T, 2> input(this->input.x, this->input.y);
        Vector<T, 2> target(this->target.x, this->target.y);

        residual[0] = weight.w * (rotation.matrix() * input + translation.vector() - target).norm();

        return true;
    }

private:
    const Point2D input;
    const Point2D target;
    const Weight weight;
};


int main(int argc, char** argv)
{
	google::InitGoogleLogging(argv[0]);

	const std::string file_path_1 = "../data/points_dragon_1.txt";
	const std::string file_path_2 = "../data/points_dragon_2.txt";
	const std::string file_path_weights = "../data/weights_dragon.txt";

    // DONE: Read data points and the weights. Define the parameters of the problem
    const auto points_1 = read_points_from_file<Point2D>(file_path_1);
    const auto points_2 = read_points_from_file<Point2D>(file_path_2);
    const auto weights = read_points_from_file<Weight>(file_path_weights);

    size_t size = points_1.size();

    // Initial values
    const double deg_initial = 0.0;
    const double tx_initial = 0.0, ty_initial = 0.0;

    double deg = deg_initial;
    double tx = tx_initial, ty = ty_initial;

	ceres::Problem problem;

	// DONE: For each weighted correspondence create one residual block
    for (unsigned int g = 0; g < size; g++) {
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<RegistrationCostFunction, 1, 1, 1, 1>(
                        new RegistrationCostFunction(points_1[g], points_2[g], weights[g])),
                nullptr, &deg, &tx, &ty
        );
    }


	ceres::Solver::Options options;
	options.max_num_iterations = 25;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;

	// DONE: Output the final values of the translation and rotation (in degree)
    std::cout << "Initial deg: " << deg_initial << "\ttx: " << tx_initial << "\tty: " << ty_initial << std::endl;
    std::cout << "Final deg: " << deg * (180.0 / M_PI) << "\ttx: " << tx << "\tty: " << ty << std::endl;

	system("pause");
	return 0;
}