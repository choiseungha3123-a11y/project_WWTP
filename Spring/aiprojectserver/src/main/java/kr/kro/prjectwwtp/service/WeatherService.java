package kr.kro.prjectwwtp.service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.domain.TmsData;
import kr.kro.prjectwwtp.persistence.WeatherRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class WeatherService {
	private final WeatherRepository weatherRepo;
	
	public TmsData findById(long id) {
		Optional<TmsData> opt = weatherRepo.findById(id);
		if(opt.isEmpty())
			return null;
		return opt.get();
	}
	
	public List<TmsData> findByTimeBetween(LocalDateTime start, LocalDateTime end) {
		return weatherRepo.findByTimeBetween(start, end);
	}
	
	public void modifyWeahter(TmsData data, double ta, double rn15m, double rn60m, double rn12h, double rnday, double hm, double td) {
		data.setTa(ta);
		data.setRn15m(rn15m);
		data.setRn60m(rn60m);
		data.setRn12h(rn12h);
		data.setRnday(rnday);
		data.setHm(hm);
		data.setTd(td);
		weatherRepo.save(data);
	}

}
