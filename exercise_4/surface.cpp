#include "utils/io.h"
#include "utils/points.h"

#include "ceres/ceres.h"
#include <cmath>


// DONE: Implement the cost function
struct SurfaceCostFunction
{
    explicit SurfaceCostFunction(const Point3D & point_) : point(point_) {}

    template<typename T>
    bool operator()(const T* const a, const T* const b, const T* const c, T* residual) const {
        // DONE: Implement the cost function
        T val = (point.x * point.x / a[0] - point.y * point.y / b[0]) / c[0];

        residual[0] = point.z - val;

        return true;
    }

private:
    const Point3D point;
};


int main(int argc, char** argv)
{
	google::InitGoogleLogging(argv[0]);

	// DONE: Read 3D surface data points and define the parameters of the problem
	const std::string file_path = "../data/points_surface.txt";
    const auto points = read_points_from_file<Point3D>(file_path);

    // Initial values
    const double a_initial = 1.0, b_initial = 1.0, c_initial = 1.0;
    double a = a_initial, b = b_initial, c = c_initial;

	ceres::Problem problem;

	// DONE: For each data point create one residual block
    for (auto & point : points) {
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SurfaceCostFunction, 1, 1, 1, 1>(
                        new SurfaceCostFunction(point)),
                nullptr, &a, &b, &c
        );
    }

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;
	
	// DONE: Output the final values of the parameters
    std::cout << "Initial a: " << a_initial << "\tb: " << b_initial << "\tc: " << c_initial << std::endl;
    std::cout << "Final a: " << a << "\tb: " << b << "\tc: " << c << std::endl;


	system("pause");
	return 0;
}