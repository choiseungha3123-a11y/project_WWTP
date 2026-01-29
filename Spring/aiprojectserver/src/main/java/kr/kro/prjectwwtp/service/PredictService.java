package kr.kro.prjectwwtp.service;

import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.persistence.MemberRepository;
import kr.kro.prjectwwtp.persistence.PredictRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class PredictService {
	private final MemberRepository memberRepo;
	private final PredictRepository predictRepo;

}
