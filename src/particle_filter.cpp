
#include "particle_filter.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

#define ASSERT (1)
#if ASSERT
#include <assert.h>
#endif

#define NUM_PARTICLE (1000)
#define NOT_EXIST (-1)
#define EPSILON (1e-5)
#define DEBUG (1)

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
#if ASSERT
  assert(sizeof(std) / sizeof(*std) == 3);
#endif

  double weight_per_particle = 1. / NUM_PARTICLE;

  std::default_random_engine rng;
  std::normal_distribution<double> noise_x(0, std[0]);
  std::normal_distribution<double> noise_y(0, std[1]);
  std::normal_distribution<double> noise_theta(0, std[2]);

  for (int i = 0; i < num_particles; ++i)
  {
    double sampled_x = x + noise_x(rng);
    double sampled_y = y + noise_y(rng);
    double sampled_theta = theta + noise_theta(rng);
    particles.push_back(Particle(i, sampled_x, sampled_y, sampled_theta, weight_per_particle));
  }

  num_particles = NUM_PARTICLE;
  is_initialized = true;
#if DEBUG
std::cout << "---init---";
for(int p=0; p<particles.size() && p < 5; ++p)
  std::cout << particles[p].x << ' ' << particles[p].y << ' ' << particles[p].theta << std::endl;
std::cout << "---init done---";
#endif
}

static inline bool is_yaw_rate_considered_zero(double yaw_rate) { return abs(yaw_rate) < EPSILON; }
void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  if(!is_yaw_rate_considered_zero(yaw_rate)){
    _prediction_with_yaw_rate(delta_t, std_pos, velocity, yaw_rate);
  }
  else{
    _prediction_without_yaw_rate(delta_t, std_pos, velocity);
  }
}

void ParticleFilter::_prediction_with_yaw_rate(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
  for (int i = 0; i < particles.size(); ++i)
  {
    double theta = particles[i].theta;
    double new_theta = theta + yaw_rate * delta_t;
    particles[i].x += velocity / yaw_rate * (sin(new_theta) - sin(theta));
    particles[i].y += velocity / yaw_rate * (cos(theta) - cos(new_theta));
    particles[i].theta = new_theta;
  }
#if DEBUG
std::cout << "---prediction_with_yaw---";
for(int p=0; p<particles.size() && p < 5; ++p)
  std::cout << particles[p].x << ' ' << particles[p].y << ' ' << particles[p].theta << std::endl;
std::cout << "---prediction_with_yaw done---";
#endif
}

void ParticleFilter::_prediction_without_yaw_rate(double delta_t, double std_pos[], double velocity)
{
  for (int i = 0; i < particles.size(); ++i)
  {
    particles[i].x += velocity * delta_t * cos(particles[i].theta);
    particles[i].y += velocity * delta_t * sin(particles[i].theta);
  }
#if DEBUG
std::cout << "---prediction_without_yaw---";
for(int p=0; p<particles.size() && p < 5; ++p)
  std::cout << particles[p].x << ' ' << particles[p].y << ' ' << particles[p].theta << std::endl;
std::cout << "---prediction_without_yaw done---";
#endif
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int i = 0; i < observations.size(); ++i)
  {
    double min_dist = std::numeric_limits<double>::max();
    int closest_landmark_id = NOT_EXIST;
    for (int p = 0; p < predicted.size(); ++p)
    {
      double distance = dist(observations[i].x, observations[i].y, predicted[p].x, predicted[p].y);
      if (distance < min_dist)
      {
        min_dist = distance;
        closest_landmark_id = predicted[p].id;
      }
    }
    observations[i].id = closest_landmark_id;
  }
#if DEBUG
std::cout << "---data association---";
for(int p=0; p<observations.size() && p < 5; ++p)
  std::cout << observations[p].id << std::endl;
std::cout << "---data association done---";
#endif
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int p = 0; p < particles.size(); ++p)
  {
    std::vector<LandmarkObs> obs_per_particle = _get_obs_per_particle(map_landmarks, particles[p], sensor_range);
    std::vector<LandmarkObs> sensed_obs_in_map_coord = _coord_transform_vehicle_to_map(observations, particles[p].x, particles[p].y, particles[p].theta);
    dataAssociation(obs_per_particle, sensed_obs_in_map_coord);
    particles[p].weight = _get_weight_of_particle(obs_per_particle, sensed_obs_in_map_coord, std_landmark);
  }
}

std::vector<LandmarkObs> ParticleFilter::_get_obs_per_particle(
    const Map &map_landmarks, const Particle &particle, double sensor_range)
{
  std::vector<LandmarkObs> obs_of_particle;
  for (int l = 0; l < map_landmarks.landmark_list.size(); ++l)
  {
    const Map::single_landmark_s &landmark = map_landmarks.landmark_list[l];
    double distance = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);
    if (distance <= sensor_range)
      obs_of_particle.push_back(LandmarkObs(landmark.id_i, landmark.x_f, landmark.y_f));
  }
  return obs_of_particle;
}

std::vector<LandmarkObs> ParticleFilter::_coord_transform_vehicle_to_map(
    const std::vector<LandmarkObs> &observations, double x, double y, double theta)
{
  std::vector<LandmarkObs> obs_in_map_coord;
  for (unsigned int j = 0; j < observations.size(); j++)
  {
    double x_map_coord = cos(theta) * observations[j].x - sin(theta) * observations[j].y + x;
    double y_map_coord = sin(theta) * observations[j].x + cos(theta) * observations[j].y + y;
    obs_in_map_coord.push_back(LandmarkObs(observations[j].id, x_map_coord, y_map_coord));
  }
  return obs_in_map_coord;
}

static inline double square(double x){ return x * x; }
static inline double squared_diff(double x, double y){ return square(x - y); }
double ParticleFilter::_get_weight_of_particle(const std::vector<LandmarkObs> &obs_per_particle, const std::vector<LandmarkObs> &sensed_obs, double std[])
{
  double weight = 1.;
  for (int s = 0; s < sensed_obs.size(); ++s)
  {
    double sensed_obs_x = sensed_obs[s].x;
    double sensed_obs_y = sensed_obs[s].y;

    double particle_obs_x = std::numeric_limits<double>::max();
    double particle_obs_y = std::numeric_limits<double>::max();
    for (int p = 0; p < obs_per_particle.size(); ++p)
    {
      if (obs_per_particle[p].id == sensed_obs[s].id)
      {
        particle_obs_x = obs_per_particle[p].x;
        particle_obs_y = obs_per_particle[p].y;
        break;
      }
    }

    double squared_diff_x = squared_diff(particle_obs_x, sensed_obs_x);
    double squared_diff_y = squared_diff(particle_obs_y, sensed_obs_y);
    double prob = (1/(2 * M_PI * std[0] * std[1])) * exp( -( squared_diff_x / (2*square(std[0])) + (squared_diff_y / (2*square(std[1]))) ) );

    if(prob == 0)
      weight *= EPSILON;
    else
      weight *= prob;
  }
  return weight;
#if DEBUG
std::cout << "---get weight of a particle---";
std::cout << weight << std::endl;
std::cout << "---get weight of a particle done---";
#endif
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
#if ASSERT
  assert(weights.size() != 0);
  double sum = 0.0;
  for (int i = 0; i < weights.size(); ++i)
    sum += weights[i];
  assert(abs(sum - 1.) < EPSILON);
#endif

  std::random_device rd;
  std::mt19937 rng(rd());
  std::discrete_distribution<> uniform_distribution(weights.begin(), weights.end());

  std::vector<Particle> sampled_particles;
  for (int i = 0; i < particles.size(); ++i)
  {
    int sampled_particle_index = uniform_distribution(rng);
    sampled_particles.push_back(particles[sampled_particle_index]);
  }

  particles = sampled_particles;
#if DEBUG
std::cout << "---resample---";
for(int p=0; p<particles.size() && p < 5; ++p)
  std::cout << particles[p].x << ' ' << particles[p].y << ' ' << particles[p].theta << std::endl;
std::cout << "---resample done---";
#endif
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
