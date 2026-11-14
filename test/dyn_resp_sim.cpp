/***************************************************************************
 *
 *   Copyright (C) 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "Pulsar/Application.h"
#include "Pulsar/DynamicResponse.h"
#include "Pulsar/ProfileColumn.h"
#include "Pulsar/Integration.h"
#include "Pulsar/Predictor.h"

#include "BoxMuller.h"
#include "random.h"
#include "strutil.h"
#include "pairutil.h"

#include <fftw3.h>
#include <cassert>
#include <cstring>
#include <cmath>
#include <ctime>

using namespace std;

//
//! Dynamic Response Simulator
//
class dyn_res_sim : public Pulsar::Application
{
public:

  //! Default constructor
  dyn_res_sim ();

  //! Seed the random number generators
  void setup ();

  //! Process the given archive
  void process (Pulsar::Archive*);

protected:

  //! Seed for the random number generators (if zero, the seed is derived from the current time)
  long seed = 0;

  //! Source of normally distributed noise
  BoxMuller gasdev;

  //! Output frequencies are reset to span the simulated bandwidth in upper sideband order
  bool reset_frequencies = false;

  //! Sampling interval in seconds
  double sampling_interval = 15.0;

  //! Number of time samples
  unsigned ntime = 256;

  //! Include the Nyquist harmonic (if false, the Nyquist harmonic is zeroed)
  bool include_Nyquist = true;

  //! Timescale of exponential decay of impulse response
  double arc_decay = 0.0;

  //! Curvature of scintillation arc
  double arc_curvature = 0.0;

  //! Arc width in pixels
  double arc_width = 0.0;

  //! Maximum delay of arc, as a fraction of the maximum delay sampled
  double arc_max_delay = 1.0;

  //! Maximum Doppler shift of arc, as a fraction of the maximum Doppler shift sampled
  double arc_max_Doppler = 0.9;

  //! Output simulated dynamic periodic spectra
  bool output_periodic_spectra = false;

  //! Max profile harmonic, as a fraction of the number of phase bins
  double max_profile_harmonic = 0.0;

  //! Asymmetry amplitude
  double asymmetry = 0.0;

  //! Phase bins per delay bin
  double phase_bin_per_delay = 0.0;

  //! Intrinsic profile is a delta function
  bool delta_profile = false;

  //! Add noise to the dynamic response, as a fraction of the rms in the response
  double dyn_resp_noise_fraction = 0.0;

  //! Add instrumental noise to the simulated dynamic periodic spectra, as a fraction of the peak harmonic amplitude
  double instrumental_noise_fraction = 0.0;

  //! Add self noise to the simulated dynamic periodic spectra, as a fraction of the flux in each phase bin
  double self_noise_fraction = 0.0;

  //! discrete scattered wave (Doppler, delay) harmonic coordinates
  std::vector<std::pair<int,int>> scattered_waves;

  //! randomize scattered wave phases
  bool scattered_waves_random_phase = false;

  //! Place a scattered wave at the origin
  bool scattered_wave_origin = true;

  //! Use a Tukey window to taper the frequency response
  double Tukey_width = 0.0;

  //! Multiply each frequency response by a random phase
  bool degenerate_phase = false;

  //! Add command line options
  void add_options (CommandLine::Menu&);

  //! Fill profile data with simulated intrinsic profile
  void generate_intrinsic_profile (Pulsar::Archive* archive);
  
  //! Generate a dynamic response based on a scintillation arc
  void generate_scintillation_arc (Pulsar::DynamicResponse* ext, double bw);

  // add a scattered wave component at "Doppler:delay"
  void add_scattered_wave (const std::string&);

  //! Generate a dynamic response based the scattered wave components
  void generate_scattered_waves (Pulsar::DynamicResponse*);

  //! Perform an in-place 2D FFT of the input wavefield, converting it to a dynamic frequency response
  void transform_wavefield (Pulsar::DynamicResponse*);

  //! Randomize the phase of each frequency response
  void randomize_phase (Pulsar::DynamicResponse*);

  //! Verify that the extension written to filename is equivalent to the first argument
  void verify_output_extension (const Pulsar::DynamicResponse* ext, const std::string& filename);

  //! Generate a periodic spectrum for each time sample of the response
  void generate_periodic_spectra (const Pulsar::DynamicResponse*, const Pulsar::Archive*);
};

dyn_res_sim::dyn_res_sim ()
  : Application ("dyn_res_sim", "Dynamic Response Simulator")
{
  Pulsar::ProfileColumn::output_floats = true;
}

void dyn_res_sim::add_options (CommandLine::Menu& menu)
{
  CommandLine::Argument* arg;

  arg = menu.add (seed, "seed", "integer");
  arg->set_help ("Seed for random number generators (default: derived from current time)");

  menu.add ("\n" "Dynamic response options:");

  arg = menu.add (sampling_interval, 't', "seconds");
  arg->set_help ("Sampling interval in seconds");

  arg = menu.add (ntime, 'n', "samples");
  arg->set_help ("Number of time samples");

  arg = menu.add (this, &dyn_res_sim::add_scattered_wave, 's', "Doppler:delay");
  arg->set_help ("add a discrete scattered wave at Doppler:delay (integer harmonic coordinates)");

  arg = menu.add (scattered_waves_random_phase, "rand");
  arg->set_help ("randomize the phases of scattered wave components");

  arg = menu.add (scattered_wave_origin, "origin");
  arg->set_help ("do not include a wavefield component at the origin");

  arg = menu.add (arc_curvature, 'c', "s^3");
  arg->set_help ("Arc curvature in seconds per square Hz");

  arg = menu.add (arc_decay, "tau", "s");
  arg->set_help ("Arc exponential decay timescale in seconds");

  arg = menu.add (arc_max_delay, 'd', "fraction");
  arg->set_help ("Arc maximum delay, as a fraction of maximum delay sampled");

  arg = menu.add (arc_max_Doppler, 'D', "fraction");
  arg->set_help ("Arc maximum Doppler shift, as a fraction of maximum Doppler shift sampled");

  arg = menu.add (arc_width, 'w', "pixels");
  arg->set_help ("Gaussian width of arc in pixels, along both Doppler and delay axes");

  arg = menu.add (Tukey_width, 'T', "flat");
  arg->set_help ("Fractional width of flat top of Tukey window");

  arg = menu.add (dyn_resp_noise_fraction, 'r', "rms");
  arg->set_help ("Standard deviation of real and imaginary parts of additional noise, relative to rms of signal");

  menu.add ("\n" "Periodic spectra options:");

  arg = menu.add (output_periodic_spectra, 'o');
  arg->set_help ("Output periodic spectrum for each time sample");

  arg = menu.add (degenerate_phase, 'x');
  arg->set_help ("Multiply each frequency response by a random phase");

  arg = menu.add (max_profile_harmonic, 'm', "fraction");
  arg->set_help ("Maximum spin harmonic, as a fraction of number of phase bins");

  arg = menu.add (asymmetry, 'a', "amplitude");
  arg->set_help ("Amplitude of asymmetry, in radians");

  arg = menu.add (phase_bin_per_delay, 'R', "ratio");
  arg->set_help ("Phase bins per delay bin");  

  arg = menu.add (delta_profile, "delta");
  arg->set_help ("Intrinsic profile is a delta function");

  arg = menu.add (include_Nyquist, "nonyq");
  arg->set_help ("do not include the Nyquist harmonic of the profile");

  arg = menu.add (instrumental_noise_fraction, 'N', "rms");
  arg->set_help ("Standard deviation of real and imaginary parts of additional noise in each harmonic, relative to peak harmonic amplitude");

  arg = menu.add (self_noise_fraction, 'S', "rms");
  arg->set_help ("Standard deviation of additional self-noise, relative to power in each phase bin");
}

void dyn_res_sim::setup ()
{
  // the current microsecond alone yields only 10^6 distinct seeds
  if (!seed)
    seed = long(time(nullptr)) * 1000000 + usec_seed();

  cerr << "dyn_res_sim::setup random number seed=" << seed << endl;

  srandom (seed);
  gasdev = BoxMuller (seed);
}

std::complex<double> random_phasor()
{
  double phase = random_double () * 2.0 * M_PI;
  return { cos(phase), sin(phase) };
}

//! Set the bandwidth and the centre frequency of each channel, such that frequency increases with channel index
void set_frequencies (Pulsar::Archive* archive, double cfreq, double bw)
{
  unsigned nchan = archive->get_nchan();
  double chanbw = bw / nchan;

  archive->set_bandwidth (bw);

  for (unsigned isub=0; isub < archive->get_nsubint(); isub++)
  {
    auto subint = archive->get_Integration(isub);
    for (unsigned ichan=0; ichan < nchan; ichan++)
      subint->set_centre_frequency (ichan, cfreq + (ichan + 0.5 - 0.5*nchan) * chanbw);
  }
}

void dyn_res_sim::process (Pulsar::Archive* archive)
{
  if (!name.empty())
    archive->set_source (name);

  archive->pscrunch();
  archive->tscrunch();

  unsigned nchan = archive->get_nchan();
  unsigned nbin = archive->get_nbin();

  // the simulated spectra are always upper sideband (see below)
  reset_frequencies = phase_bin_per_delay || archive->get_bandwidth() < 0;

  // the dynamic response spans ntime contiguous intervals, starting at the start of the input archive
  MJD start_time = archive->start_time();

  if (phase_bin_per_delay)
  {
    cerr << "dyn_res_sim::process nchan=" << nchan << " nbin=" << nbin << endl;

    // each phase bin is 1 fake millisecond
    double phase_bin_time_interval = 1e-3;
    double period = nbin * phase_bin_time_interval;

    // get rid of the phase predictor
    if (archive->has_model())
      delete archive->get_model();
    auto subint = archive->get_Integration(0);
    subint->set_folding_period(period);

    double time_interval = phase_bin_time_interval * phase_bin_per_delay;
    double bw = 1.0 / time_interval;
    cerr << "dyn_res_sim::process simulated bandwidth=" << bw << " Hz" << endl;
    archive->set_bandwidth(bw*1e-6); // stored in MHz
  }

  double cfreq = archive->get_centre_frequency();

  // Radio frequency increases with channel index in all simulated output, regardless of the
  // sideband of the input archive.  A causal response then lies at +ve delays along the
  // Fourier conjugate of the channel index, as assumed by pycyc (which uses only |bw|).
  double bw = fabs(archive->get_bandwidth());

  Reference::To<Pulsar::DynamicResponse> ext = new Pulsar::DynamicResponse;
  ext->set_centre_frequency(cfreq);
  ext->set_bandwidth(bw);

  // as assumed by dsp::DynamicFilter, response itime spans min + interval * [itime, itime+1],
  // where interval = (max - min) / ntime
  ext->set_minimum_epoch(start_time);
  ext->set_maximum_epoch(start_time + ntime * sampling_interval);

  ext->set_nchan(nchan);
  ext->set_ntime(ntime);
  ext->set_npol(1);
  ext->resize_data();

  if (scattered_waves.size() != 0)
    generate_scattered_waves (ext);
  else
    generate_scintillation_arc (ext, bw);

  if (max_profile_harmonic || delta_profile)
    generate_intrinsic_profile (archive);

  Reference::To<Pulsar::Archive> clone = archive->clone();
  clone->pscrunch();
  clone->fscrunch();
  clone->tscrunch();
  if (reset_frequencies)
    set_frequencies (clone, cfreq, bw);
  clone->add_extension(ext);

  std::string filename = "dyn_resp_sim.fits";
  clone->unload(filename);

  verify_output_extension (ext, filename);

  if (output_periodic_spectra)
  {
    generate_periodic_spectra (ext, archive);
  }
}

void copy (float* amps, vector<complex<double>>& profile)
{
  unsigned nbin = profile.size();
  double real_power = 0.0;
  double imag_power = 0.0;

  for (unsigned ibin=0; ibin < nbin; ibin++)
  {
    double re = profile[ibin].real();
    double im = profile[ibin].imag();

    real_power += re*re;
    imag_power += im*im;

    amps[ibin] = re;
  }

  if (real_power > 0 && imag_power >= real_power * 1e-20)
  {
    throw Error (InvalidState, "copy", "power real=%f imag=%f", real_power, imag_power);
  }
}

void dyn_res_sim::generate_intrinsic_profile (Pulsar::Archive* archive)
{
  archive->pscrunch();
  archive->tscrunch();

  unsigned nbin = archive->get_nbin();
  unsigned nchan = archive->get_nchan();

  if (delta_profile)
  {
    cerr << "dyn_res_sim::generate_intrinsic_profile delta function nchan=" << nchan << endl;
    Pulsar::Integration* subint = archive->get_Integration(0);
    subint->zero();

    unsigned ipol = 0;
    for (unsigned ichan = 0; ichan < nchan; ichan++)
    {
      subint->set_weight(ichan, 1.0);
      auto profile = subint->get_Profile(ipol,ichan);
      auto f_amps = profile->get_amps();
      f_amps[0] = 1.0;
    }
    return;
  }

  vector<std::complex<double>> profile (nbin, 0.0);

  // maximum dynamic range supported by single-precision floating point profile amplitudes 
  // in PSRFITS files, as determined by trial-and-error using psrplot -U -c val=I
  double log10_max = 0.0;
  double log10_min = -8.0;

  unsigned ibin_max = 0.5 * max_profile_harmonic * nbin;

  double log10_curvature = ( log10_min - log10_max ) / (ibin_max * ibin_max);

  // DC bin
  profile[0] = 0.0;

  // when nbin is even, the Nyquist harmonic (ibin = nbin/2) remains zero
  for (unsigned ibin=1; ibin < (nbin+1)/2; ibin++)
  {
    double log10_amp = log10_max + log10_curvature * ibin * ibin;
    double amp = pow(10.0, log10_amp);

    profile[ibin] = amp;

    if (asymmetry)
    {
      // rotate the phase of each harmonic by up to -asymmetry radians, without changing its amplitude
      double im_amp = 1.0 / (1.0 + exp(-0.5*ibin/double(ibin_max)));
      profile[ibin] = std::polar(amp, -asymmetry*im_amp);
    }

    profile[nbin-ibin] = std::conj(profile[ibin]);
  }

  auto fftinout = reinterpret_cast<fftw_complex*>( profile.data() );
  auto plan = fftw_plan_dft_1d (nbin, fftinout, fftinout, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);

  Pulsar::Integration* subint = archive->get_Integration(0);
  unsigned ipol = 0;
  for (unsigned ichan = 0; ichan < nchan; ichan++)
  {
    auto f_amps = subint->get_Profile(ipol,ichan)->get_amps();
    copy(f_amps,profile);
  }
}

void dyn_res_sim::verify_output_extension (const Pulsar::DynamicResponse* ext, const std::string& filename)
{
  const char* method = "dyn_res_sim::verify_output_extension";

  cerr << method << " get expected extension data" << endl;
  auto data = ext->get_data().data();
  unsigned nchan = ext->get_nchan();
  unsigned ntime = ext->get_ntime();

  Reference::To<Pulsar::Archive> test = Pulsar::Archive::load(filename);
  auto test_ext = test->get<Pulsar::DynamicResponse>();
  if (!test_ext)
    throw Error (InvalidState, method, filename + " does not contain a DynamicResponse extension");

  if (test_ext->get_nchan() != nchan || test_ext->get_ntime() != ntime || test_ext->get_npol() != ext->get_npol())
    throw Error (InvalidState, method, "loaded ntime=%u nchan=%u npol=%u != expected ntime=%u nchan=%u npol=%u",
                 test_ext->get_ntime(), test_ext->get_nchan(), test_ext->get_npol(), ntime, nchan, ext->get_npol());

  unsigned errors = 0;

  // header parameters are stored as double-precision values in text
  auto differ = [](double loaded, double expected) { return fabs(loaded - expected) > 1e-12 * fabs(expected); };

  if (differ (test_ext->get_centre_frequency(), ext->get_centre_frequency()))
  {
    cerr << method << " centre frequency loaded=" << test_ext->get_centre_frequency() << " != " << ext->get_centre_frequency() << endl;
    errors ++;
  }

  if (differ (test_ext->get_bandwidth(), ext->get_bandwidth()))
  {
    cerr << method << " bandwidth loaded=" << test_ext->get_bandwidth() << " != " << ext->get_bandwidth() << endl;
    errors ++;
  }

  // epochs are stored as a double-precision number of days, with a resolution of ~10 microseconds
  double epoch_tolerance = 1e-4; // seconds

  if (fabs((test_ext->get_minimum_epoch() - ext->get_minimum_epoch()).in_seconds()) > epoch_tolerance)
  {
    cerr << method << " minimum epoch loaded=" << test_ext->get_minimum_epoch() << " != " << ext->get_minimum_epoch() << endl;
    errors ++;
  }

  if (fabs((test_ext->get_maximum_epoch() - ext->get_maximum_epoch()).in_seconds()) > epoch_tolerance)
  {
    cerr << method << " maximum epoch loaded=" << test_ext->get_maximum_epoch() << " != " << ext->get_maximum_epoch() << endl;
    errors ++;
  }

  if (test_ext->get_impulse_pos() != ext->get_impulse_pos() || test_ext->get_impulse_neg() != ext->get_impulse_neg())
  {
    cerr << method << " impulse pos:neg loaded=" << test_ext->get_impulse_pos() << ":" << test_ext->get_impulse_neg()
         << " != " << ext->get_impulse_pos() << ":" << ext->get_impulse_neg() << endl;
    errors ++;
  }

  cerr << method << " get loaded extension data" << endl;
  auto test_data = test_ext->get_data().data();

  // the response is stored as double-precision complex values
  unsigned data_errors = 0;
  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    for (unsigned itime=0; itime < ntime; itime++)
    {
      unsigned idx = itime*nchan + ichan;
      auto diff = test_data[idx] - data[idx];
      if (abs(diff) > 1e-12)
      {
        if (data_errors < 10)
          cerr << "ichan=" << ichan << " itime=" << itime << " " << data[idx] << " - " << test_data[idx] << " = " << diff << endl;
        data_errors ++;
      }
    }
  }

  if (errors || data_errors)
    throw Error (InvalidState, method, "%u header parameters and %u response values in %s differ from expected values",
                 errors, data_errors, filename.c_str());
}

void add_response (complex<double>* data, unsigned jomega, unsigned jtau, unsigned ntime, unsigned nchan, double amplitude, double arc_width)
{
  if (!arc_width)
  {
    data[jomega*nchan + jtau] += amplitude * random_phasor();
    return;
  }

  // compute the Gaussian out to where the amplitude falls to 10^-9
  int fizzle = ceil(arc_width * sqrt(9 * log(10.0)));
  // cerr << "add_response compute Gaussian out to " << fizzle << " pixels from arc" << endl;

  // signed dimensions, so that negative coordinates are wrapped correctly
  int nom = ntime;
  int ntau = nchan;

  for (int iom=int(jomega)-fizzle; iom <= int(jomega)+fizzle; iom++)
    for (int itau=int(jtau)-fizzle; itau <= int(jtau)+fizzle; itau++)
    {
      double dist_om = (double(iom) - double(jomega)) / arc_width;
      double dist_tau = (double(itau) - double(jtau)) / arc_width;
      double amp = exp( -dist_om*dist_om -dist_tau*dist_tau );

      unsigned kom = ((iom % nom) + nom) % nom;
      unsigned ktau = ((itau % ntau) + ntau) % ntau;

      data[kom*nchan + ktau] += amplitude * amp * random_phasor();
    }
}

void Tukey (vector<double>& window, double fraction_flat)
{
  unsigned ndat = window.size();
  unsigned transition_end = ndat * (1-fraction_flat) / 2;

  cerr << "Tukey: ndat=" << ndat << " transition_end=" << transition_end << endl;

  for (unsigned idat=0; idat < (ndat+1)/2; idat++)
  {
    double amp = 1.0;

    // offset by half a sample, so that the first and last samples are not weighted by zero
    if (idat < transition_end)
      amp = 0.5 * (1 - cos(M_PI*(idat+0.5)/transition_end));

    window[idat] = window[ndat-idat-1] = amp;
  }
}






// perform an in-place 2D FFT

/*! 
In principle, we wish to perform a forward FFT along the delay axis and a backward FFT
along the differential Doppler delay axis.  This could be achieved by reversing the
elements along differential Doppler delay axis (index j -> -j modulo ntime), as done in
generate_scattered_waves.  However, when the phases are random, it doesn't matter (at
least, as long as only the dynamic frequency response is used from this point onward,
and there is no need to return to the delay-Doppler wavefield).
*/
void dyn_res_sim::transform_wavefield (Pulsar::DynamicResponse* ext)
{
  unsigned nchan = ext->get_nchan();
  unsigned ntime = ext->get_ntime();
  auto data = ext->get_data().data();

  // record the extent of the impulse response along the delay axis before it is transformed
  // (as in generate_periodic_spectra, the upper half of the delay axis is treated as -ve delays)
  unsigned impulse_pos = 0;
  unsigned impulse_neg = 0;

  for (unsigned itime=0; itime < ntime; itime++)
    for (unsigned ichan=1; ichan < nchan; ichan++)
      if (data[itime*nchan + ichan] != 0.0)
      {
        if (ichan < (nchan+1)/2)
          impulse_pos = std::max(impulse_pos, ichan);
        else
          impulse_neg = std::max(impulse_neg, nchan-ichan);
      }

  // both additive noise and tapering in the frequency domain spread the impulse response over all delays
  if (dyn_resp_noise_fraction > 0.0 || Tukey_width)
  {
    impulse_pos = (nchan-1)/2;
    impulse_neg = nchan/2;
  }

  ext->set_impulse_pos(impulse_pos);
  ext->set_impulse_neg(impulse_neg);

  auto fftin = reinterpret_cast<fftw_complex*>(data);
  auto plan = fftw_plan_dft_2d(ntime, nchan, fftin, fftin, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);

  unsigned ndat = nchan * ntime;
  double total_power = 0.0;
  for (unsigned idat=0; idat < ndat; idat++)
    total_power += norm(data[idat]);

  double rms = sqrt(total_power/ndat);
  cerr << "dyn_res_sim::transform_wavefield normalizing by rms of dynamic response=" << rms << endl;
  for (unsigned idat=0; idat < ndat; idat++)
    data[idat] /= rms;

  if (dyn_resp_noise_fraction > 0.0)
  {
    cerr << "dyn_res_sim::transform_wavefield adding noise with fractional rms=" << dyn_resp_noise_fraction << endl;
    for (unsigned idat=0; idat < ndat; idat++)
    {
      data[idat] += dyn_resp_noise_fraction * complex<double>(gasdev(),gasdev());
    }
  }

  if (Tukey_width)
  {
    vector<double> window (nchan);
    Tukey (window, Tukey_width);

    double power_before = 0.0;
    double power_after = 0.0;

    for (unsigned itime=0; itime < ntime; itime++)
      for (unsigned ichan=0; ichan < nchan; ichan++)
      {
        power_before += norm(data[itime*nchan + ichan]);
        data[itime*nchan + ichan] *= window[ichan];
        power_after += norm(data[itime*nchan + ichan]);
      }

    // preserve the rms of the dynamic response
    double rescale = sqrt(power_before / power_after);
    for (unsigned idat=0; idat < ndat; idat++)
      data[idat] *= rescale;
  }

  if (degenerate_phase)
  {
    randomize_phase(ext);
  }
}

void dyn_res_sim::randomize_phase (Pulsar::DynamicResponse* ext)
{
  unsigned nchan = ext->get_nchan();
  unsigned ntime = ext->get_ntime();
  auto data = ext->get_data().data();

  for (unsigned itime=0; itime < ntime; itime++)
  {
    auto phasor = random_phasor();

    complex<double> sum = 0.0;

    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      unsigned idx = itime*nchan + ichan;
      data[idx] *= phasor;
      sum += data[idx];
    }

#if MINIMIZE_DC_PHASE

    phasor = conj(sum) / abs(sum);

    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      unsigned idx = itime*nchan + ichan;
      data[idx] *= phasor;
    }

#else

    if (itime > 0)
    {
      auto s0 = data + (itime-1)*nchan;
      auto s1 = data + itime*nchan;

      sum = 0.0;
      for (unsigned ichan=0; ichan < nchan; ichan++)
      {
        sum += s0[ichan] * conj(s1[ichan]);
      }

      phasor = sum / abs(sum);
      for (unsigned ichan=0; ichan < nchan; ichan++)
      {
        s1[ichan] *= phasor;
      }
    }

#endif
  }
}

void dyn_res_sim::add_scattered_wave (const std::string& arg)
{
  auto coords = fromstring<std::pair<int,int>> (arg);
  cerr << "dyn_res_sim::add_scattered_wave coords=" << coords << endl;
  scattered_waves.push_back(coords);
}

void dyn_res_sim::generate_scattered_waves(Pulsar::DynamicResponse* ext)
{
  unsigned nchan = ext->get_nchan();
  unsigned ntime = ext->get_ntime();

  cerr << "dyn_res_sim::generate_scattered_waves dimensions=" << ntime << ":" << nchan << endl;

  auto data = ext->get_data().data();

  for (unsigned ichan=0; ichan < nchan; ichan++)
    for (unsigned itime=0; itime < ntime; itime++)
      data[itime*nchan + ichan] = 0;

  if (scattered_wave_origin)
    data[0] = 1.0;

  for (auto wave: scattered_waves)
  {
    int itime = wave.first;
    int ichan = wave.second;

    if (2*abs(itime) > int(ntime) || 2*abs(ichan) > int(nchan))
      throw Error (InvalidParam, "dyn_res_sim::generate_scattered_waves",
                   "Doppler:delay=%d:%d exceeds the maximum harmonic coordinates=%u:%u",
                   itime, ichan, ntime/2, nchan/2);

    // reverse the time axis ... see note about reversal at dyn_res_sim::transform_wavefield
    int jtime = (int(ntime)-itime) % int(ntime);
    int jchan = (int(nchan)+ichan) % int(nchan);

    cerr << "dyn_res_sim::generate_scattered_waves coords=" << jtime << ":" << jchan << endl;

    complex<double> value = 1.0;
    if (scattered_waves_random_phase)
      value = random_phasor();

    data[jtime*nchan + jchan] = value;
  
  }

  transform_wavefield (ext);
}

void dyn_res_sim::generate_scintillation_arc (Pulsar::DynamicResponse* ext, double bw)
{
  unsigned nchan = ext->get_nchan();
  unsigned ntime = ext->get_ntime();

  if (arc_max_delay <= 0.0 || arc_max_delay > 1.0)
    throw Error (InvalidParam, "dyn_res_sim::generate_scintillation_arc",
                 "arc maximum delay fraction=%lf must be greater than 0 and no greater than 1", arc_max_delay);

  auto data = ext->get_data().data();

  for (unsigned ichan=0; ichan < nchan; ichan++)
    for (unsigned itime=0; itime < ntime; itime++)
      data[itime*nchan + ichan] = 0;

  // sampling interval along delay axis, in seconds (bw is always positive; see dyn_res_sim::process)
  double delta_tau = 1e-6/bw;
  // maximum positive delay
  double max_tau = 0.5 * nchan * delta_tau; 

  // time spanned by response
  double time_span = ntime * sampling_interval;

  // sampling interval along Doppler shift axis, in Hz
  double delta_omega = 1.0/time_span;
  // maximum positive Doppler shift
  double max_omega = 0.5 * ntime * delta_omega;

  double curvature = arc_curvature;
  if (curvature == 0)
  {
    cerr << "dyn_res_sim::process setting arc curvature to span " << arc_max_Doppler*100.0 << "% of Doppler axis at maximum delay" << endl;
    double span_omega = arc_max_Doppler * max_omega;
    curvature = arc_max_delay * max_tau / (span_omega * span_omega);
  }

  cerr << "dyn_res_sim::process arc curvature = " << curvature << " s^3" << endl;

  double decay = arc_decay;
  if (decay == 0)
  {
    double default_decay = 0.1;
    cerr << "dyn_res_sim::process setting decay time scale to " << default_decay*100.0 << "% of maximum delay" << endl;
    decay = arc_max_delay * max_tau * default_decay;
  }

  cerr << "dyn_res_sim::process decay time scale = " << decay << " s" << endl;

  unsigned iomega = 0;
  unsigned nomega = ntime / 2;

  unsigned itau = 0;
  unsigned ntau = arc_max_delay * nchan / 2;

  cerr << "dyn_res_sim::process nomega=" << nomega << " ntau=" << ntau << endl;

  bool f_of_omega = true;

  double omega = 0.0;
  double tau = 0.0;
  unsigned jtau = 0;
  unsigned jomega = 0;

  while (iomega < nomega && itau < ntau)
  {
    if (f_of_omega)
    {
      omega = iomega * delta_omega;
      tau = curvature * omega*omega;
      jtau = lround(tau / delta_tau);
      jomega = iomega;

      // when the step in tau is larger than the sampling interval (i.e. a delay is skipped)
      if (jtau > itau)
      {
        cerr << "switch to function of tau when iomega=" << iomega << " and itau=" << itau << endl;
        f_of_omega = false;
      }
      else
      {
        iomega ++;
        itau = jtau + 1;
      }
    }

    if (!f_of_omega)
    {
      cerr << " " << itau;
      tau = itau * delta_tau;
      omega = sqrt(tau/curvature);
      jtau = itau;
      jomega = lround(omega / delta_omega);
      if (jomega >= nomega)
        break;

      itau ++;
      iomega = jomega;
    }

    double amplitude = exp(- tau/decay);
    // cout << jtau << " " << amplitude << endl;

    add_response (data, jomega, jtau, ntime, nchan, amplitude, arc_width);

    if (jomega > 0)
      add_response (data, ntime-jomega, jtau, ntime, nchan, amplitude, arc_width);
  }

  cerr << "loop finished with iomega=" << iomega << " and itau=" << itau << endl;

  transform_wavefield (ext);
}

//! Generate a periodic spectrum for each time sample of the response
void dyn_res_sim::generate_periodic_spectra (const Pulsar::DynamicResponse* ext, const Pulsar::Archive* archive) try
{
  auto data = ext->get_data().data();
  unsigned nchan = ext->get_nchan();
  unsigned ntime = ext->get_ntime();

  assert(nchan == archive->get_nchan());
  unsigned nbin = archive->get_nbin();

  // always positive; see dyn_res_sim::process
  double bw_MHz = ext->get_bandwidth();
  double chanbw_Hz = bw_MHz * 1e6 / nchan;

  MJD start_time = ext->get_minimum_epoch();

  Reference::To<Pulsar::Archive> prototype = archive->clone();
  prototype->pscrunch();
  prototype->fscrunch();
  prototype->tscrunch();

  cerr << "dyn_res_sim::generate_periodic_spectra prototype profile computed" << endl;

  auto subint = prototype->get_Integration(0);
  double folding_period = subint->get_folding_period();
  double spin_frequency = 1.0 / folding_period;

  const float* f_amps = prototype->get_Profile(0,0,0)->get_amps();
  vector<std::complex<double>> intrinsic_spectrum (nbin);
  for (unsigned ibin=0; ibin<nbin; ibin++)
    intrinsic_spectrum[ibin] = f_amps[ibin];

  auto fftin = reinterpret_cast<fftw_complex*>(intrinsic_spectrum.data());
  auto plan = fftw_plan_dft_1d (nbin, fftin, fftin, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);

  // only when nbin is even is there a Nyquist harmonic
  bool has_Nyquist = (nbin % 2 == 0);

  if (has_Nyquist && !include_Nyquist)
    intrinsic_spectrum[nbin/2] = 0.0;

  // number of harmonics computed (the remaining harmonics are set by Hermitian symmetry)
  unsigned nharm = (has_Nyquist && !include_Nyquist) ? nbin/2 : nbin/2 + 1;

  // verify that the intrinsic spectrum has the expected Hermiticity
  double total_power = 0;
  double diff_power = 0;
  double max_power = 0;

  for (unsigned ibin=1; ibin < (nbin+1)/2; ibin++)
  {
    auto diff = intrinsic_spectrum[ibin] - conj(intrinsic_spectrum[nbin - ibin]);
    double power = norm(intrinsic_spectrum[ibin]);
    if (power > max_power)
      max_power = power;
    total_power += power;
    diff_power += norm(diff);
  }

  if (diff_power > total_power * 1e-20)
  {
    cerr << "Intrinsic spectrum does not appear to be Hermitian power=" << total_power << " diff power=" << diff_power << endl;
  }

  // standard deviation of the real and imaginary parts of the noise in each harmonic
  double instrumental_noise_rms = instrumental_noise_fraction * sqrt(max_power);

  // White noise with the above standard deviation in the real and imaginary parts of each harmonic
  // is, after the unnormalized backward FFT performed below, equivalent to white noise in each
  // phase bin with a variance equal to nbin times the variance of each complex-valued harmonic.
  // It is added to each phase bin, so that the DC and Nyquist harmonics require no special treatment.
  double instrumental_noise_rms_per_bin = instrumental_noise_rms * sqrt(2.0 * nbin);

  if (self_noise_fraction > 0.0)
    cerr << "dyn_res_sim::generate_periodic_spectra adding self noise with fractional rms=" << self_noise_fraction << endl;

  vector<std::complex<double>> temp (nbin);
  vector<std::complex<double>> profile (nbin);
  fftin = reinterpret_cast<fftw_complex*>( temp.data() );
  auto fftout = reinterpret_cast<fftw_complex*>( profile.data() );
  auto bin_plan = fftw_plan_dft_1d (nbin, fftin, fftout, FFTW_BACKWARD, FFTW_ESTIMATE);

  vector<std::complex<double>> frequency_response (nchan);
  vector<std::complex<double>> impulse_response (nchan);
  vector<std::complex<double>> shifted_impulse_response (nchan);

  fftin = reinterpret_cast<fftw_complex*>( frequency_response.data() );
  fftout = reinterpret_cast<fftw_complex*>( impulse_response.data() );
  auto bwd_plan = fftw_plan_dft_1d (nchan, fftin, fftout, FFTW_BACKWARD, FFTW_ESTIMATE);

  // re-use frequency response for the shifted frequency response
  fftin = reinterpret_cast<fftw_complex*>( shifted_impulse_response.data() );
  fftout = reinterpret_cast<fftw_complex*>( frequency_response.data() );
  auto fwd_plan = fftw_plan_dft_1d (nchan, fftin, fftout, FFTW_FORWARD, FFTW_ESTIMATE);

  vector<std::complex<double>> cyclic_spectrum (nbin * nchan);

  prototype = archive->clone();
  prototype->pscrunch();

  if (reset_frequencies)
    set_frequencies (prototype, ext->get_centre_frequency(), bw_MHz);

  // Normalize the forward FFT of the intrinsic profile and the backward FFT of each cyclic spectrum.
  // As the dynamic response has unit rms, the flux scale of the intrinsic profile is preserved.
  double scalefac = 1.0 / nbin;

  for (unsigned itime=0; itime < ntime; itime++)
  {
    // initialize a cyclic spectrum that is nchan copies of the intrinsic spectrum
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      memcpy(cyclic_spectrum.data() + ichan*nbin, intrinsic_spectrum.data(), nbin * sizeof(complex<double>));
    }

    // copy from the dynamic response to the input frequency response
    memcpy(frequency_response.data(), data + itime * nchan, sizeof(complex<double>) * nchan);
    fftw_execute(bwd_plan);

    // impulse_response now contains the inverse FFT of the frequency response

    // the combination of fwd and bwd plans results in scaling by nchan
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      impulse_response[ichan] /= nchan;
    }

    for (unsigned ibin=0; ibin < nharm; ibin++)
    {
      for (int sign: {-1, 1})
      {
        double alpha = ibin * spin_frequency;
        double slope = -sign * M_PI * alpha / chanbw_Hz; // 2pi * alpha/2

        shifted_impulse_response = impulse_response;
        for (unsigned ichan=1; ichan < nchan; ichan++)
        {
          // treat the upper half of the array as -ve delays
          int jchan = (ichan < (nchan+1)/2) ? ichan : int(ichan) - int(nchan);
          double phase = slope * double(jchan) / nchan;
          complex<double> phasor (cos(phase), sin(phase));
          shifted_impulse_response[ichan] *= phasor;
        }

        fftw_execute(fwd_plan);
        // frequency_response now contains the shifted frequency response function

        for (unsigned ichan=0; ichan < nchan; ichan++)
        {
          if (sign == -1)
            frequency_response[ichan] = conj(frequency_response[ichan]);

          auto spectrum = cyclic_spectrum.data() + ichan*nbin;

          // S'(nu;alpha) = H(nu + alpha/2) H^*(nu - alpha/2) S(nu;alpha)
          spectrum[ibin] *= frequency_response[ichan];

          // after multiplying by both H(nu + alpha/2) and H^*(nu - alpha/2)
          if (sign == 1)
          {
            if (2*ibin == nbin)
            {
              // The Nyquist harmonic of a real-valued signal sampled at nbin phase bins is the sum of
              // the harmonics at +nbin/2 and -nbin/2, which are complex conjugates of each other;
              // i.e. S'(nu;+alpha) + S'(nu;-alpha) = 2 Re[ H(nu + alpha/2) H^*(nu - alpha/2) S(nu;alpha) ]
              // where the factor of 2 is already included in the Nyquist harmonic of the intrinsic spectrum
              spectrum[ibin] = spectrum[ibin].real();
            }
            else if (ibin > 0)
              spectrum[nbin-ibin] = conj(spectrum[ibin]);  // Hermitian spectrum
          }
        }
      }
    }

    Reference::To<Pulsar::Archive> output = prototype->clone();
    auto subint = output->get_Integration(0);
    unsigned ipol = 0;

    // each periodic spectrum is integrated over a single sampling interval of the dynamic response
    subint->set_epoch (start_time + (itime + 0.5) * sampling_interval);
    subint->set_duration (sampling_interval);

    for (unsigned ichan=0; ichan < nchan; ichan++) try
    {
      memcpy(temp.data(), cyclic_spectrum.data() + ichan*nbin, nbin * sizeof(complex<double>));
      fftw_execute(bin_plan);
      // profile now contains the periodic spectrum for ichan

      // self noise is proportional to the scattered flux in each phase bin of each channel,
      // and is independent between phase bins, channels, and time samples
      if (self_noise_fraction > 0.0)
        for (unsigned ibin=0; ibin < nbin; ibin++)
          profile[ibin] *= 1.0 + self_noise_fraction*gasdev();

      if (instrumental_noise_rms_per_bin > 0.0)
        for (unsigned ibin=0; ibin < nbin; ibin++)
          profile[ibin] += instrumental_noise_rms_per_bin*gasdev();

      auto f_amps = subint->get_Profile(ipol,ichan)->get_amps();
      copy(f_amps, profile);
      subint->get_Profile(ipol,ichan)->scale(scalefac);
    }
    catch (Error& error)
    {
      throw error += "dyn_res_sim::generate_periodic_spectra ichan=" + tostring(ichan);
    }

    string filename = "periodic_spectrum_" + stringprintf("%05d",itime) + ".ar";
    cerr << "unloading " << filename << endl;
    output->unload(filename);
  }

  fftw_destroy_plan(bwd_plan);
  fftw_destroy_plan(fwd_plan);
  fftw_destroy_plan(bin_plan);
}
catch (Error& error)
{
  throw error += "dyn_res_sim::generate_periodic_spectra";
}
/*!

  The standard C/C++ main function simply calls Application::main

*/

int main (int argc, char** argv)
{
  // the random number generators are seeded in dyn_res_sim::setup
  dyn_res_sim program;
  return program.main (argc, argv);
}

