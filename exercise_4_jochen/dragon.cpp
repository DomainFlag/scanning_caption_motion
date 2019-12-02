#include "utils/io.h"
#include "utils/points.h"

#include "ceres/ceres.h"
#include <math.h>


// TODO: Implement the cost function
struct RegistrationCostFunction
{
    RegistrationCostFunction(const Point2D& point_1_, const Point2D& point_2_, const Weight weight_)
            : point_1(point_1_)
            , point_2(point_2_)
            , weight(weight_)
    {
    }

    template<typename T>
    bool operator()(const T* const rad, const T* const tx, const T* const ty, T* residual) const
    {
        auto x_diff = T(point_1.x)*cos(-rad[0]) + T(point_1.y)*sin(-rad[0]) + tx[0] - point_2.x;
        auto y_diff = T(point_1.y)*cos(-rad[0]) - T(point_1.x)*sin(-rad[0]) + ty[0] - point_2.y;

        residual[0] = T(weight.w) * sqrt(x_diff * x_diff + y_diff * y_diff) ;

        return true;
    }

private:
    const Point2D point_1;
    const Point2D point_2;
    const Weight weight;

};


int main(int argc, char** argv)
{
	google::InitGoogleLogging(argv[0]);

	// TODO: Read data points and the weights. Define the parameters of the problem
	const std::string file_path_1 = "../data/points_dragon_1.txt";
	const std::string file_path_2 = "../data/points_dragon_2.txt";
	const std::string file_path_weights = "../data/weights_dragon.txt";

    const auto points_1 = read_points_from_file<Point2D>(file_path_1);
    const auto points_2 = read_points_from_file<Point2D>(file_path_2);
    const auto weights =  read_points_from_file<Weight>(file_path_weights);

    const double rad_initial = 0.0;
    const double tx_initial = 0.0;
    const double ty_initial = 0.0;

    double rad = rad_initial;
    double tx = tx_initial;
    double ty = ty_initial;


    ceres::Problem problem;

	// TODO: For each weighted correspondence create one residual block
    for (int i = 0; i<points_1.size(); i++)
    {
        auto& point_1 = points_1[i];
        auto& point_2 = points_2[i];
        auto& weight = weights[i];

        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<RegistrationCostFunction, 1, 1, 1, 1>(
                        new RegistrationCostFunction(point_1, point_2, weight)),
                nullptr, &rad, &tx, &ty
        );
    }

	ceres::Solver::Options options;
	options.max_num_iterations = 25;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;

	// TODO: Output the final values of the translation and rotation (in degree)
    std::cout << "Final   deg: " << (rad / M_PI * 180.0) << "\ttx: " << tx << "\tty: " << ty << std::endl;

	return 0;
}