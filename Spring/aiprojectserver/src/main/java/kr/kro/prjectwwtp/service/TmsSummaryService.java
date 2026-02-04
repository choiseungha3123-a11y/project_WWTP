package kr.kro.prjectwwtp.service;

import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.domain.TmsSummary;
import kr.kro.prjectwwtp.persistence.TmsSummaryRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class TmsSummaryService {
	private final TmsSummaryRepository repo;
	
	public List<Date> getFakeDatesList() {
		List<Date> retList = new ArrayList<Date>();
		List<TmsSummary> summaries = repo.findAll();
		int checkNum = 2600;
		
		TmsSummary pre = null;
		boolean first = true;
		for(TmsSummary summary : summaries) {
			pre = summary;
			if(first)
			{
				first = false;
				continue;
			}
			if( pre.getCount() + summary.getCount() >= checkNum &&
					ChronoUnit.DAYS.between(pre.getTime().toInstant(), summary.getTime().toInstant()) == 1) {
				// 하루전 날짜와의 합계가 checkNum 이상인 경우
				retList.add(summary.getTime());
				}
		}
		return retList;
	}
}
