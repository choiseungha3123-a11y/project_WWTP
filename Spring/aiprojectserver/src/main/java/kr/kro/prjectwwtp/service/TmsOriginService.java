package kr.kro.prjectwwtp.service;

import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.persistence.TmsOriginRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class TmsOriginService {
	private final TmsOriginRepository tmsOriginRepo;

}
