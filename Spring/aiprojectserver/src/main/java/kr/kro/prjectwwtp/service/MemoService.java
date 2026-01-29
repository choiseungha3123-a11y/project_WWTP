package kr.kro.prjectwwtp.service;

import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.persistence.MemberRepository;
import kr.kro.prjectwwtp.persistence.MemoRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class MemoService {
	private final MemberRepository memberRepo;
	private final MemoRepository memoRepo;
	

}
