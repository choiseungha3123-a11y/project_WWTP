package kr.kro.prjectwwtp.service;

import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.domain.WeatherApiLog;
import kr.kro.prjectwwtp.persistence.WeatherAPILogRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class WeatherAPILogService {
private final WeatherAPILogRepository logRepo;
	
	public void addWeatherAPILog(String logType, int originSize, int returnSize, int modifySize, String requestURI, String errorMsg) {
		// 로그 추가
		logRepo.save(WeatherApiLog.builder()
							.logType(logType)
							.originSize(originSize)
							.returnSize(returnSize)
							.modifySize(modifySize)
							.requestURI(requestURI)
							.errorMsg(errorMsg)
							.build());
	}
}
