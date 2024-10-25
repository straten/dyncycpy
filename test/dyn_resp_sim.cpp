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
#include "random.h"
#include "strutil.h"
#include "pairutil.h"

#include <fftw3.h>
#include <cassert>
#include <cstring>

using namespace std;

//
//! Dynamic Response Simulator
//
class dyn_res_sim : public Pulsar::Application
{
public:

  //! Default constructor
  dyn_res_sim ();

  //! Process the given archive
  void process (Pulsar::Archive*);

protected:

  //! Sampling interval in seconds
  double sampling_interval = 15.0;

  //! Number of time samples
  unsigned ntime = 256;

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

  //! Max profile harmonic, as a fraction of the number of phase bins
  double max_profile_harmonic = 0.0;

  //! Output dynamic periodic spectra
  bool output_periodic_spectra = false;

  //! discrete scattered wave (Doppler, delay) harmonic coordinates
  std::pair<unsigned,unsigned> scattered_wave = {0,0};

  double Tukey_width = 0.0;

  //! Add command line options
  void add_options (CommandLine::Menu&);

  //! Fill profile data with simulated intrinsic profile
  void generate_intrinsic_profile (Pulsar::Archive* archive);
  
  //! Generate a dynamic response based on a scintillation arc
  void generate_scintillation_arc (Pulsar::DynamicResponse* ext, double bw);

  //! Generate a dynamic response based on a single scattered wave component
  void generate_scattered_wave (Pulsar::DynamicResponse*);

  //! Perform an in-place 2D FFT of the input wavefield, converting it to a dynamic frequency response
  void transform_wavefield (Pulsar::DynamicResponse*);
    
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

  menu.add ("\n" "General options:");

  arg = menu.add (sampling_interval, 't', "seconds");
  arg->set_help ("Sampling interval in seconds");

  arg = menu.add (ntime, 'n', "samples");
  arg->set_help ("Number of time samples");

  arg = menu.add (max_profile_harmonic, 'm', "fraction");
  arg->set_help ("maximum profile harmonic, as a fraction of number of phase bins");

  arg = menu.add (scattered_wave, 's', "Doppler:delay");
  arg->set_help ("Doppler,delay harmonic coordinates of discrete scattered wave");

  arg = menu.add (arc_curvature, 'c', "s^3");
  arg->set_help ("Arc curvature in seconds per square Hz");

  arg = menu.add (arc_decay, "tau", "s");
  arg->set_help ("Arc exponential decay timescale in seconds");

  arg = menu.add (arc_max_delay, 'd', "fraction");
  arg->set_help ("Arc maximum delay, as a fraction of maximum delay sampled");

  arg = menu.add (arc_max_Doppler, 'D', "fraction");
  arg->set_help ("Arc maximum Doppler shift, as a fraction of maximum Doppler shift sampled");

  arg = menu.add (arc_width, 'w', "pixels");
  arg->set_help ("Arc width in pixels along Doppler axis");

  arg = menu.add (Tukey_width, 'T', "flat");
  arg->set_help ("Fractional width of flat top of Tukey window");

  arg = menu.add (output_periodic_spectra, 'o');
  arg->set_help ("Output periodic spectrum for each time sample");
}

std::complex<double> random_phasor()
{
  double phase = random_double () * 2.0 * M_PI;
  return { cos(phase), sin(phase) };
}

void dyn_res_sim::process (Pulsar::Archive* archive)
{
  if (!name.empty())
    archive->set_source (name);

  unsigned nchan = archive->get_nchan();
  unsigned npol = archive->get_npol();

  double cfreq = archive->get_centre_frequency();
  double bw = archive->get_bandwidth();
  double chanbw = bw / nchan;

  double minfreq = cfreq - 0.5 * (bw - chanbw);
  double maxfreq = cfreq + 0.5 * (bw - chanbw);

  Reference::To<Pulsar::DynamicResponse> ext = new Pulsar::DynamicResponse;
  ext->set_minimum_frequency(minfreq);
  ext->set_maximum_frequency(maxfreq);

  ext->set_nchan(nchan);
  ext->set_ntime(ntime);
  ext->set_npol(1);
  ext->resize_data();

  if (scattered_wave.first != 0 || scattered_wave.second != 0)
    generate_scattered_wave (ext);
  else
    generate_scintillation_arc (ext, bw);

  if (max_profile_harmonic)
    generate_intrinsic_profile (archive);

  Reference::To<Pulsar::Archive> clone = archive->clone();
  clone->pscrunch();
  clone->fscrunch();
  clone->tscrunch();
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

  assert (imag_power < real_power * 1e-20);
}

void dyn_res_sim::generate_intrinsic_profile (Pulsar::Archive* archive)
{
  archive->pscrunch();
  archive->tscrunch();

  unsigned nbin = archive->get_nbin();
  unsigned nchan = archive->get_nchan();

  vector<std::complex<double>> profile (nbin, 0.0);

  // maximum dynamic range supported by single-precision floating point profile amplitudes 
  // in PSRFITS files, as determined by trial-and-error using psrplot -U -c val=I
  double log10_max = 0.0;
  double log10_min = -8.0;

  unsigned ibin_min = 1;
  unsigned ibin_max = 0.5 * max_profile_harmonic * nbin;

  double log10_slope = ( log10_min - log10_max ) / (ibin_max - ibin_min);

  // DC bin
  profile[0] = 0.0;

  for (unsigned ibin=1; ibin < nbin/2; ibin++)
  {
    double log10_amp = log10_max + (ibin - ibin_min) * log10_slope;
    profile[nbin-ibin] = profile[ibin] = pow(10.0, log10_amp);
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
  cerr << "dyn_res_sim::verify_output_extension get expected extension data" << endl;
  auto data = ext->get_data().data();
  unsigned nchan = ext->get_nchan();
  unsigned ntime = ext->get_ntime();

  Reference::To<Pulsar::Archive> test = Pulsar::Archive::load(filename);
  auto test_ext = test->get<Pulsar::DynamicResponse>();
  cerr << "dyn_res_sim::verify_output_extension get loaded extension data" << endl;
  auto test_data = test_ext->get_data().data();
  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    for (unsigned itime=0; itime < ntime; itime++)
    {
      unsigned idx = itime*nchan + ichan;
      auto diff = test_data[idx] - data[idx];
      if (abs(diff) > 1e-3)
      {
        cerr << "ichan=" << ichan << " itime=" << itime << " " << data[idx] << " - " << test_data[idx] << " = " << diff << endl;
      }
    }
  }
}

void add_response (complex<double>* data, unsigned jomega, unsigned jtau, unsigned ntime, unsigned nchan, double amplitude, double arc_width)
{
  data[jomega*nchan + jtau] += amplitude * random_phasor();

  if (arc_width)
  {
    for (unsigned iom=0; iom < ntime; iom++)
    {
      double dist = (double(iom) - double(jomega)) / arc_width;
      double amp = exp( -dist*dist );
      data[iom*nchan + jtau] += amplitude * amp * random_phasor();
    }
  }
}

void Tukey (vector<double>& window, double fraction_flat)
{
  unsigned ndat = window.size();
  unsigned transition_end = ndat * (1-fraction_flat) / 2;
  double frequency = M_PI/transition_end;

  cerr << "Tukey: ndat=" << ndat << " transition_end=" << transition_end << endl;

  float denom_start = 2*transition_end;
  float denom_end = 2*transition_end;

  for (unsigned idat=0; idat<ndat/2; idat++)
  {
    double amp = 1.0;
    if (idat < transition_end)
      amp = 0.5 * (1 - cos(idat*frequency));

    window[idat] = window[ndat-idat-1] = amp;
  }
}

// perform an in-place 2D FFT

/*! 
In principle, we wish to perform a forward FFT along the delay axis and a backward FFT
along the differential Doppler delay axis.  This could be achieved by complex conjugating
and reversing the elements along differential Doppler delay axis.  However, since the
phases are random, it doesn't matter (at least, as long as only the dynamic frequency 
response is used from this point onward, and there is no need to return to the
delay-Doppler wavefield).
*/
void dyn_res_sim::transform_wavefield (Pulsar::DynamicResponse* ext)
{
  unsigned nchan = ext->get_nchan();
  unsigned ntime = ext->get_ntime();
  auto data = ext->get_data().data();

  auto fftin = reinterpret_cast<fftw_complex*>(data);
  auto plan = fftw_plan_dft_2d(ntime, nchan, fftin, fftin, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);

  if (Tukey_width)
  {
    vector<double> window (nchan);
    Tukey (window, Tukey_width);

    for (unsigned itime=0; itime < ntime; itime++)
      for (unsigned ichan=0; ichan < nchan; ichan++)
        data[itime*nchan + ichan] *= window[ichan];
  }
}

void dyn_res_sim::generate_scattered_wave(Pulsar::DynamicResponse* ext)
{
  unsigned nchan = ext->get_nchan();
  unsigned ntime = ext->get_ntime();

  auto data = ext->get_data().data();

  for (unsigned ichan=0; ichan < nchan; ichan++)
    for (unsigned itime=0; itime < ntime; itime++)
      data[itime*nchan + ichan] = 0;

  unsigned itime = scattered_wave.first;
  unsigned ichan = scattered_wave.second;

  data[0] = 1.0;

  /* Reflect Doppler coordinate because a single forward 2D FFT 
     is performed by transform_wavefield, when it should be forward 
     along delay-to-frequency backward along Doppler-to-time. */
  unsigned jtime = ntime - itime;
  data[jtime*nchan + ichan] = 1.0;

  transform_wavefield (ext);
}

void dyn_res_sim::generate_scintillation_arc (Pulsar::DynamicResponse* ext, double bw)
{
  unsigned nchan = ext->get_nchan();
  unsigned ntime = ext->get_ntime();

  auto data = ext->get_data().data();

  for (unsigned ichan=0; ichan < nchan; ichan++)
    for (unsigned itime=0; itime < ntime; itime++)
      data[itime*nchan + ichan] = 0;

  // sampling interval along delay axis, in seconds
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
    cerr << "dyn_res_sim::process setting arc curvature to span " << arc_max_Doppler*100.0 << "\% of Doppler axis at maximum delay" << endl;
    double span_omega = arc_max_Doppler * max_omega;
    curvature = arc_max_delay * max_tau / (span_omega * span_omega);
  }

  cerr << "dyn_res_sim::process arc curvature = " << curvature << " s^3" << endl;

  double decay = arc_decay;
  if (decay == 0)
  {
    double default_decay = 0.1;
    cerr << "dyn_res_sim::process setting decay time scale to " << default_decay*100.0 << "\% of maximum delay" << endl;
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
      jtau = tau / delta_tau;
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
      tau = itau * delta_tau;
      omega = sqrt(tau/curvature);
      jtau = itau;
      jomega = omega / delta_omega;
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
void dyn_res_sim::generate_periodic_spectra (const Pulsar::DynamicResponse* ext, const Pulsar::Archive* archive)
{
  auto data = ext->get_data().data();
  unsigned nchan = ext->get_nchan();
  unsigned ntime = ext->get_ntime();

  assert(nchan == archive->get_nchan());
  unsigned nbin = archive->get_nbin();

  double bw_MHz = archive->get_bandwidth();
  double chanbw_Hz = bw_MHz * 1e6 / nchan;

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

  // verify that the intrinsic spectrum has the expected Hermiticity
  double power = 0;
  double diff_power = 0;
  for (unsigned ibin=1; ibin < nbin/2; ibin++)
  {
    auto diff = intrinsic_spectrum[ibin] - conj(intrinsic_spectrum[nbin - ibin]);
    power += norm(intrinsic_spectrum[ibin]);
    diff_power += norm(diff);
  }

  if (diff_power > power * 1e-20)
  {
    cerr << "Intrinsic spectrum does not appear to be Hermitian power=" << power << " diff power=" << diff_power << endl;
  }

  vector<std::complex<double>> temp (nbin);
  vector<std::complex<double>> profile (nbin);
  fftin = reinterpret_cast<fftw_complex*>( temp.data() );
  auto fftout = reinterpret_cast<fftw_complex*>( profile.data() );
  plan = fftw_plan_dft_1d (nbin, fftin, fftout, FFTW_BACKWARD, FFTW_ESTIMATE);

  vector<std::complex<double>> frequency_response (nchan);
  vector<std::complex<double>> impulse_response (nchan);
  vector<std::complex<double>> shifted_impulse_response (nchan);

  fftin = reinterpret_cast<fftw_complex*>( frequency_response.data() );
  fftout = reinterpret_cast<fftw_complex*>( impulse_response.data() );
  auto fwd_plan = fftw_plan_dft_1d (nchan, fftin, fftout, FFTW_BACKWARD, FFTW_ESTIMATE);

  // re-use frequency reponse for the shifted frequency response
  fftin = reinterpret_cast<fftw_complex*>( shifted_impulse_response.data() );
  fftout = reinterpret_cast<fftw_complex*>( frequency_response.data() );
  auto bwd_plan = fftw_plan_dft_1d (nchan, fftin, fftout, FFTW_FORWARD, FFTW_ESTIMATE);

  vector<std::complex<double>> cyclic_spectrum (nbin * nchan);

  prototype = archive->clone();
  prototype->pscrunch();

  for (unsigned itime=0; itime < ntime; itime++)
  {
    // initialize a cyclic spectrum that is nchan copies of the profile FFT
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      memcpy(cyclic_spectrum.data() + ichan*nbin, intrinsic_spectrum.data(), nbin * sizeof(complex<double>));
    }

    // copy from the dynamic response to the input frequency response
    memcpy(frequency_response.data(), data + itime * nchan, sizeof(complex<double>) * nchan);
    fftw_execute(fwd_plan);
    // impulse_response now contains the FFT of the frequency response

    for (unsigned ibin=1; ibin < nbin/2; ibin++)
    {
      for (int sign: {-1, 1})
      {
        double alpha = ibin * spin_frequency;
        double slope = sign * M_PI * alpha / chanbw_Hz; // 2pi * alpha/2

        shifted_impulse_response = impulse_response;
        for (unsigned ichan=1; ichan < nchan; ichan++)
        {
          double phase = slope * double(ichan) / nchan;
          complex<double> phasor (cos(phase), sin(phase));
          shifted_impulse_response[ichan] *= phasor;
        }

        fftw_execute(bwd_plan);
        // frequency_response now contains the shifted frequency response function

        for (unsigned ichan=0; ichan < nchan; ichan++)
        {
          if (sign == 1)
            frequency_response[ichan] = conj(frequency_response[ichan]);

          auto spectrum = cyclic_spectrum.data() + ichan*nbin;

          spectrum[ibin] *= frequency_response[ichan];
          spectrum[nbin-ibin] = conj(spectrum[ibin]);  // Hermitian spectrum
        }
      }
    }

    Reference::To<Pulsar::Archive> output = prototype->clone();
    auto subint = output->get_Integration(0);
    unsigned ipol = 0;

    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      memcpy(temp.data(), cyclic_spectrum.data() + ichan*nbin, nbin * sizeof(complex<double>));
      fftw_execute(plan);
      // profile now contains the periodic spectrum for ichan

      auto f_amps = subint->get_Profile(ipol,ichan)->get_amps();
      copy(f_amps, profile);
    }

    string filename = "periodic_spectrum_" + stringprintf("%05d",itime) + ".ar";
    cerr << "unloading " << filename << endl;
    output->unload(filename);
  }

  fftw_destroy_plan(fwd_plan);
  fftw_destroy_plan(bwd_plan);
  fftw_destroy_plan(plan);
}

/*!

  The standard C/C++ main function simply calls Application::main

*/

int main (int argc, char** argv)
{
  // seeds the random number generator with the current microsecond
  random_init ();

  dyn_res_sim program;
  return program.main (argc, argv);
}

